PYTHONPATH=. python3 scripts/train_img2img.py \
                    --category CMP \
                    --load_size 154 \
                    --image_size 128 \
                    --image_nc 3    \
                    --attr_size 10  \
                    --des_net   descriptor_img2img    \
                    --gen_net   generator_resnet128     \
                    $@