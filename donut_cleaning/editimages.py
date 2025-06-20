import cv2
import numpy as np
from pathlib import Path

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
SEARCH_WORD = "00/268, randhawa chowk, tirupati-095744"
LABELS_DIR  = Path("./dehado_cropped_dataset/labels")
IMAGES_DIR  = LABELS_DIR.parent / "images"
IMAGE_EXT   = ".jpg"
# ────────────────────────────────────────────────────────────────────────────────

class ImageEditor:
    def __init__(self, image_path):
        self.image_path = Path(image_path)
        self.window_name = f"Edit: {self.image_path.name}"
        self.orig   = cv2.imread(str(self.image_path))
        if self.orig is None:
            raise RuntimeError(f"Could not load '{self.image_path}'")
        self.canvas = self.orig.copy()  # we'll white-out crops here
        # state
        self.snippets     = []  # each: dict(orig_img, x,y,w,h)
        self.drawing      = False
        self.moving       = False
        self.resizing     = False
        self.current_box  = None
        self.selected_idx = -1
        self.move_start   = (0,0)
        self.resize_anchor= (0,0)
        self.CORNER_TH    = 8  # px tolerance

    def composite_without_boxes(self):
        out = self.canvas.copy()
        h_img, w_img = out.shape[:2]
        for sn in self.snippets:
            resized = cv2.resize(sn['orig_img'], (sn['w'], sn['h']), interpolation=cv2.INTER_AREA)
            x,y,w,h = sn['x'],sn['y'],sn['w'],sn['h']
            x0,y0 = max(0,x), max(0,y)
            x1,y1 = min(w_img, x+w), min(h_img, y+h)
            dx0,dy0 = x0-x, y0-y
            dx1,dy1 = dx0+(x1-x0), dy0+(y1-y0)
            out[y0:y1, x0:x1] = resized[dy0:dy1, dx0:dx1]
        return out

    def overlay_with_boxes(self):
        out = self.composite_without_boxes()
        for i,sn in enumerate(self.snippets):
            x,y,w,h = sn['x'],sn['y'],sn['w'],sn['h']
            col = (0,255,0) if i==self.selected_idx else (0,0,255)
            cv2.rectangle(out, (x,y), (x+w,y+h), col, 2)
        return out

    def hit_corner(self, sn, mx, my):
        corners = {
            'tl': (sn['x'],        sn['y']),
            'br': (sn['x']+sn['w'], sn['y']+sn['h'])
        }
        for name,(cx,cy) in corners.items():
            if abs(mx-cx)<=self.CORNER_TH and abs(my-cy)<=self.CORNER_TH:
                return name
        return None

    def on_mouse(self, evt, mx, my, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN:
            # 1) resize?
            for idx,sn in enumerate(self.snippets):
                c = self.hit_corner(sn, mx, my)
                if c:
                    self.resizing = True
                    self.selected_idx = idx
                    if c == 'tl':
                        self.resize_anchor = (sn['x']+sn['w'], sn['y']+sn['h'])
                    else:
                        self.resize_anchor = (sn['x'], sn['y'])
                    return
            # 2) move?
            for idx,sn in enumerate(self.snippets):
                if sn['x']<=mx<=sn['x']+sn['w'] and sn['y']<=my<=sn['y']+sn['h']:
                    self.moving = True
                    self.selected_idx = idx
                    self.move_start = (mx,my)
                    return
            # 3) new box?
            if len(self.snippets) < 2:
                self.drawing = True
                self.current_box = [mx, my, mx, my]

        elif evt == cv2.EVENT_MOUSEMOVE:
            if self.drawing and self.current_box is not None:
                self.current_box[2:] = [mx, my]
                tmp = self.overlay_with_boxes()
                x0,y0,x1,y1 = self.current_box
                cv2.rectangle(tmp, (x0,y0), (x1,y1), (255,0,0), 1)
                cv2.imshow(self.window_name, tmp)

            elif self.moving and self.selected_idx >= 0:
                dx, dy = mx-self.move_start[0], my-self.move_start[1]
                sn = self.snippets[self.selected_idx]
                sn['x'] += dx; sn['y'] += dy
                self.move_start = (mx,my)
                cv2.imshow(self.window_name, self.overlay_with_boxes())

            elif self.resizing and self.selected_idx >= 0:
                ax, ay = self.resize_anchor
                x0, x1 = sorted((ax, mx))
                y0, y1 = sorted((ay, my))
                sn = self.snippets[self.selected_idx]
                sn.update({'x':x0, 'y':y0, 'w':x1-x0, 'h':y1-y0})
                cv2.imshow(self.window_name, self.overlay_with_boxes())

        elif evt == cv2.EVENT_LBUTTONUP:
            if self.drawing and self.current_box is not None:
                x0,y0,x1,y1 = self.current_box
                x0,x1 = sorted((x0,x1)); y0,y1 = sorted((y0,y1))
                if x1>x0 and y1>y0:
                    crop = self.orig[y0:y1, x0:x1].copy()
                    self.canvas[y0:y1, x0:x1] = 255  # white-out
                    self.snippets.append({
                        'orig_img': crop,
                        'x': x0, 'y': y0,
                        'w': x1-x0, 'h': y1-y0
                    })
                self.drawing = False
                self.current_box = None
                cv2.imshow(self.window_name, self.overlay_with_boxes())

            if self.moving:
                self.moving = False
                self.selected_idx = -1

            if self.resizing:
                self.resizing = False
                self.selected_idx = -1

    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.on_mouse)
        cv2.imshow(self.window_name, self.overlay_with_boxes())
        print(f"\n--- Editing {self.image_path.name} ---")
        print(" Draw: drag blank space (max 2 snippets).")
        print(" Move: drag inside.  Resize: drag corner.")
        print(" Save & overwrite: 's'    Skip: 'q' or Esc\n")

        while True:
            key = cv2.waitKey(20) & 0xFF
            if key in (ord('q'), 27):
                print(f"Skipped {self.image_path.name}\n")
                break
            if key == ord('s'):
                result = self.composite_without_boxes()
                cv2.imwrite(str(self.image_path), result)
                print(f"Overwrote {self.image_path}\n")
                break

        cv2.destroyWindow(self.window_name)


def main():
    txt_files = list(LABELS_DIR.rglob("*.txt"))
    print(f"Scanning {len(txt_files)} label files for '{SEARCH_WORD}'…\n")

    for txt in txt_files:
        try:
            with txt.open("r", encoding="utf-8", errors="ignore") as f:
                if SEARCH_WORD in f.read().lower():
                    img_path = IMAGES_DIR / (txt.stem + IMAGE_EXT)
                    print("image path: ", img_path)
                    if img_path.exists():
                        editor = ImageEditor(img_path)
                        editor.run()
                    else:
                        print(f"No image for {txt.name}: looked at {img_path}")
        except Exception as e:
            print(f"⚠️  Error processing {txt}: {e}")

    print("All done.")

if __name__ == "__main__":
    main()
