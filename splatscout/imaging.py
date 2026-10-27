"""发给模型的图: 裁剪 / 叠图 / 白底抠图 / 编码. 全部纯 numpy+cv2, 不碰 GPU."""
import cv2
import numpy as np

from . import config

PALETTE = [(60, 60, 230), (60, 180, 60), (0, 150, 255), (200, 110, 240), (210, 210, 80),
           (240, 130, 130), (30, 190, 210), (160, 100, 40)]


def to_jpeg(img, long_side=config.THUMB_LONG, q=85):
    h, w = img.shape[:2]
    s = long_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    assert ok
    return buf.tobytes()


def stamp(img, text, org=(6, 22), scale=0.7, color=(255, 255, 255)):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = org
    cv2.rectangle(img, (x - 6, y - th - 8), (x + tw + 6, y + 8), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)
    return img


def bbox_of(mask):
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def pad_box(box, shape, pad=config.CROP_PAD, min_side=config.CROP_MIN_SIDE):
    """外扩 + 最小边 + 裁到画幅内. box=(x0,y0,x1,y1)."""
    H, W = shape[:2]
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    bw, bh = max((x1 - x0) * (1 + 2 * pad), min_side), max((y1 - y0) * (1 + 2 * pad), min_side)
    x0, x1 = int(max(0, cx - bw / 2)), int(min(W, cx + bw / 2))
    y0, y1 = int(max(0, cy - bh / 2)), int(min(H, cy + bh / 2))
    return x0, y0, x1, y1


def crop(img, box):
    x0, y0, x1, y1 = box
    return img[y0:y1, x0:x1].copy()


def overlay(img, masks_labels, alpha=0.45, lw=2):
    """masks_labels: [(mask bool HxW, label str, color idx)] -> 染色 + 轮廓 + 质心标号."""
    out = img.copy()
    for mask, label, ci in masks_labels:
        if not mask.any():
            continue
        c = PALETTE[ci % len(PALETTE)]
        out[mask] = (alpha * np.array(c) + (1 - alpha) * out[mask]).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, (255, 255, 255), lw)
    for mask, label, ci in masks_labels:
        if not mask.any():
            continue
        ys, xs = np.nonzero(mask)
        x, y = int(np.median(xs)), int(np.median(ys))
        stamp(out, label, (max(6, x - 20), max(22, y)), 0.8, PALETTE[ci % len(PALETTE)])
    return out


def contour_only(img, mask, color=(40, 40, 230), lw=3, tint=0.2):
    out = img.copy()
    if mask.any():
        out[mask] = (tint * np.array(color) + (1 - tint) * out[mask]).astype(np.uint8)
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, lw)
    return out


def cutout(img, mask, pad=0.1):
    """白底抠图: 掩码外刷白, 裁到掩码框外扩 pad."""
    out = np.full_like(img, 255)
    out[mask] = img[mask]
    box = bbox_of(mask)
    if box is None:
        return out
    return crop(out, pad_box(box, img.shape, pad=pad, min_side=128))


def draw_box(img, box, color=(0, 230, 255), lw=2):
    """在图上画一个矩形框 (原图坐标 x0,y0,x1,y1); 用来把模型画的框描给人看."""
    x0, y0, x1, y1 = [int(v) for v in box]
    cv2.rectangle(img, (x0, y0), (max(x1 - 1, x0), max(y1 - 1, y0)), color, lw)
    return img


def whiten_outside(img, mask):
    out = np.full_like(img, 255)
    out[mask] = img[mask]
    return out


def dilate(mask, px):
    if px <= 0:
        return mask
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k).astype(bool)


def resize_mask(mask, wh):
    if mask.shape[1] == wh[0] and mask.shape[0] == wh[1]:
        return mask.astype(bool)
    return cv2.resize(mask.astype(np.uint8), wh, interpolation=cv2.INTER_NEAREST).astype(bool)


def contact_sheet(items, side=config.THUMB_SIDE, cols=3):
    """拼图: items = [(jpeg bytes, label)], 每格 side x side 白底, 左上角印标签. 返回 jpeg bytes."""
    items = list(items)[:max(1, cols * 2)] if len(items) > cols * 2 else list(items)
    n = max(1, len(items))
    cols = min(cols, n)
    rows = (n + cols - 1) // cols
    sheet = np.full((rows * side, cols * side, 3), 255, np.uint8)
    for i, (jpg, label) in enumerate(items):
        arr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        if arr is None:
            continue
        h, w = arr.shape[:2]
        s = (side - 8) / max(h, w)
        arr = cv2.resize(arr, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA)
        r, c = divmod(i, cols)
        y0, x0 = r * side + 4, c * side + 4
        sheet[y0:y0 + arr.shape[0], x0:x0 + arr.shape[1]] = arr
        stamp(sheet, str(label), (x0 + 2, y0 + 18), 0.6)
        cv2.rectangle(sheet, (c * side, r * side), (c * side + side - 1, r * side + side - 1), (200, 200, 200), 1)
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    assert ok
    return buf.tobytes()


def erode(mask, px):
    if px <= 0:
        return mask.astype(bool)
    k = np.ones((2 * px + 1, 2 * px + 1), np.uint8)
    return cv2.erode(mask.astype(np.uint8), k).astype(bool)
