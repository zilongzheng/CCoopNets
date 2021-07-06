MODE=$1


if [[ $MODE != "extended" && $MODE != "base" ]]; then
    echo "Must specify mode as \"base\" or \"extended\""
    exit 1
fi

if [[ $MODE == "base" ]]; then 
    ZIP_FILE=CMP_facade_DB_base.zip
    CONTENT_DIR=base
fi

if [[ $MODE == "extended" ]]; then
    ZIP_FILE=CMP_facade_DB_extended.zip
    CONTENT_DIR=extended
fi

URL=https://cmp.felk.cvut.cz/~tylecr1/facade/$ZIP_FILE
mkdir -p ./datasets/CMP
wget -N $URL -O ./datasets/CMP/$ZIP_FILE
cd ./datasets/CMP
unzip -o $ZIP_FILE
rm $ZIP_FILE
mkdir -p $CONTENT_DIR/photos
mkdir -p $CONTENT_DIR/facade
mv $CONTENT_DIR/*.jpg $CONTENT_DIR/photos/
mv $CONTENT_DIR/*.png $CONTENT_DIR/facade/
cd ../..

