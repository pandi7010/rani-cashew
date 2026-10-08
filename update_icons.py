from PIL import Image
import numpy as np

def make_trans(path_in, path_out):
    img = Image.open(path_in).convert('RGBA').resize((22, 22), Image.Resampling.LANCZOS)
    arr = np.array(img)
    mask = (arr[:,:,0].astype(int) + arr[:,:,1].astype(int) + arr[:,:,2].astype(int)) < 350
    arr[mask, 3] = 0
    Image.fromarray(arr).save(path_out)

make_trans('/home/phoenix/.gemini/antigravity-ide/brain/337c32a9-d497-47f9-bb3b-ce48b3efb6b8/btn_test_new_1789493757529.jpg', '/home/phoenix/Documents/1/cashew_gsn/CNN/assets/btn_test.png')
make_trans('/home/phoenix/.gemini/antigravity-ide/brain/337c32a9-d497-47f9-bb3b-ce48b3efb6b8/btn_gear_new_1789493769693.jpg', '/home/phoenix/Documents/1/cashew_gsn/CNN/assets/btn_gear.png')
print("Icons updated!")
