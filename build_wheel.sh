
if [ -d "dist" ]; then
rm -rf ./dist/
fi

mkdir dist/
python3 -m build
#python3 -m twine upload  dist/* #uncomment for uploading
