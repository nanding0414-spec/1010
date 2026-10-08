from pathlib import Path
import tempfile
import unittest
import numpy as np
from PIL import Image
from tools.probe_sam3_resolution import select_probe_sample, load_probe_regions


class ProbeSelectionTests(unittest.TestCase):
    def test_skip_small_without_shrinking_and_pin_worker_sample(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images = root/'data/VOC2012/JPEGImages'
            masks = root/'storage/pseudo_labels/voc/sam3_active'
            split = root/'data/VOC2012/ImageSets/Segmentation/trainaug.txt'
            images.mkdir(parents=True); masks.mkdir(parents=True); split.parent.mkdir(parents=True)
            split.write_text('small\nlarge\n')
            for name, edge in [('small',10), ('large',80)]:
                Image.new('RGB',(120,120)).save(images/f'{name}.jpg')
                label = np.zeros((120,120),np.uint8)
                label[10:10+edge,10:10+edge]=1
                Image.fromarray(label).save(masks/f'{name}.png')
            image, mask, _, record = select_probe_sample(root)
            self.assertEqual(image.stem,'large')
            self.assertEqual(record['region_count'],1)
            self.assertEqual(len(record['skipped']),1)
            _, regions, _ = load_probe_regions(image,mask)
            self.assertEqual(int(regions[0].sum()),6400)
            selected = select_probe_sample(root,image,mask)
            self.assertEqual(selected[0],image)
            with self.assertRaisesRegex(ValueError,'Explicit sample'):
                select_probe_sample(root,images/'small.jpg',masks/'small.png')
            with self.assertRaises(FileNotFoundError):
                select_probe_sample(root,span_index={})
            # Invalid labels must fail, not silently select the next image.
            Image.new('L',(120,120),30).save(masks/'small.png')
            with self.assertRaisesRegex(ValueError,'Class-ID'):
                select_probe_sample(root)

    def test_explicit_mask_requires_image(self):
        with self.assertRaisesRegex(ValueError,'explicit --image'):
            select_probe_sample(Path('.'), mask=Path('mask.png'))


if __name__ == '__main__':
    unittest.main()
