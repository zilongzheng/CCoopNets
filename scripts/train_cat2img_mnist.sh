PYTHONPATH=. python3 scripts/train_cat2img.py \
                    --image_size 28 \
                    --image_nc 1    \
                    --attr_size 10  \
                    --des_net   descriptor_mnist    \
                    --gen_net   generator_mnist     \
                    $@