"""Diagnostic artifacts for the experimental active proposal mode."""

import json

import cv2
import numpy as np
from PIL import Image, ImageDraw

from tools.visualize_proposals import proposal_color, select_proposals


def class_overlay(rgb, masks, scores, class_ids, classes):
    """Stable class colors across before/after panels; labels retain proposal IDs."""
    overlay = rgb.copy()
    occupied = np.zeros(rgb.shape[:2], dtype=bool)
    order = select_proposals(masks, scores)
    label_boxes = []
    for index in order:
        color = np.asarray(proposal_color(int(class_ids[index]) - 1))
        visible = masks[index] & ~occupied
        overlay[visible] = np.rint(rgb[visible] * .65 + color * .35).astype(np.uint8)
        occupied |= masks[index]
    for index in reversed(order):
        binary = masks[index].astype(np.uint8)
        color = proposal_color(int(class_ids[index]) - 1)
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2, cv2.LINE_AA)
        distance = cv2.distanceTransform(np.pad(binary, 1), cv2.DIST_L2, 3)[1:-1, 1:-1]
        y, x = np.unravel_index(distance.argmax(), distance.shape)
        name = classes[int(class_ids[index]) - 1]['name']
        label = f'#{index + 1} {name} {scores[index]:.3f}'
        (width, height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, .4, 1)
        x = max(0, min(int(x), rgb.shape[1] - width - 2))
        y = max(height + 1, min(int(y), rgb.shape[0] - baseline - 1))
        # Competing masks often have identical centers; separate their captions.
        for shift in [0] + [sign * (height + baseline + 5) * hop
                            for hop in range(1, 9) for sign in (1, -1)]:
            candidate_y = y + shift
            box = (x, candidate_y - height - 2, x + width + 2, candidate_y + baseline + 2)
            if box[1] < 0 or box[3] >= rgb.shape[0]:
                continue
            if any(box[0] < b[2] and box[2] > b[0] and box[1] < b[3] and box[3] > b[1]
                   for b in label_boxes):
                continue
            y = candidate_y
            label_boxes.append(box)
            break
        else:
            continue  # Contours remain visible when no uncluttered label fits.
        cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .4, (255, 255, 255), 1, cv2.LINE_AA)
    return overlay


def save_active_result(folder, rgb, masks, scores, result):
    report = result['report']
    classes = report['classes']
    # Extend the ordinary cache format while retaining masks/scores compatibility.
    np.savez_compressed(folder / 'proposals.npz', masks=masks.astype(np.uint8), scores=scores,
                        class_ids=result['class_ids'])
    final = class_overlay(rgb, masks, scores, result['class_ids'], classes)
    Image.fromarray(final).save(folder / 'final_class_overlay.png')
    # Class-colored proposals without any semantic competition or calibration.
    width, height = rgb.shape[1], rgb.shape[0]
    comparison = Image.new('RGB', (width * 2, height + 28), 'white')
    draw = ImageDraw.Draw(comparison)
    for i, (title, panel) in enumerate(zip(('Original', 'Proposals (class colors)'), (rgb, final))):
        comparison.paste(Image.fromarray(panel), (i * width, 28))
        draw.text((i * width + 6, 7), title, fill='black')
    comparison.save(folder / 'comparison_classes.png')
    palette = np.zeros((len(classes) + 1, 3), dtype=np.uint8)
    for row in classes:
        palette[row['id']] = proposal_color(row['id'] - 1)
    # Vocabulary report doubles as a class-color legend and preview audit.
    chart = Image.new('RGB', (760, 52 + len(classes) * 25), 'white')
    draw = ImageDraw.Draw(chart)
    draw.text((12, 10), 'Low-resolution presence / active classes (color = class ID)', fill='black')
    active = set(report['active_class_ids'])
    presence = report['preview_presence']
    for i, row in enumerate(classes):
        y = 42 + i * 25
        draw.rectangle((12, y, 26, y + 14), fill=tuple(palette[row['id']]))
        draw.text((34, y), f'{row["id"]:02d} {row["name"]}', fill='black')
        score = presence[i] if presence is not None else None
        if score is not None:
            draw.rectangle((280, y, 280 + round(260 * score), y + 14), fill=(74, 126, 183))
        status = 'KEEP' if row['id'] in active else 'DROP'
        caption = f'{score:.3f}  {status}' if score is not None else f'not previewed: {status}'
        draw.text((555, y), caption, fill='black')
    chart.save(folder / 'preview_classes.png')
    report = dict(report)
    report['proposal_class_ids'] = result['class_ids'].tolist()
    (folder / 'active_report.json').write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
