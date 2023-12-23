cd model_zoo
mkdir Release
cd Release
wget https://huggingface.co/datasets/Shengjie/WildCamera/resolve/main/checkpoint/wild_camera_all.pth?download=true
mv wild_camera_all.pth?download=true wild_camera_all.pth
wget https://huggingface.co/datasets/Shengjie/WildCamera/resolve/main/checkpoint/wild_camera_gsv.pth?download=true
mv wild_camera_gsv.pth?download=true wild_camera_gsv.pth
cd ..
mkdir swin_transformer
cd swin_transformer
wget https://huggingface.co/datasets/Shengjie/WildCamera/resolve/main/checkpoint/swin_large_patch4_window7_224_22k.pth?download=true
mv swin_large_patch4_window7_224_22k.pth?download=true swin_large_patch4_window7_224_22k.pth
cd ..
