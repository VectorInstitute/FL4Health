data_path="fl4health/utils/datasets"

mkdir ${data_path}/HAM10000
mv ${data_path}/HAM10000_images_part_1.zip ${data_path}/HAM10000/
mv ${data_path}/HAM10000_images_part_2.zip ${data_path}/HAM10000/
unzip ${data_path}/HAM10000/HAM10000_images_part_1.zip -d ${data_path}/HAM10000/
unzip ${data_path}/HAM10000/HAM10000_images_part_2.zip -d ${data_path}/HAM10000/
rm ${data_path}/HAM10000/HAM10000_images_part_1.zip
rm ${data_path}/HAM10000/HAM10000_images_part_2.zip
mv ${data_path}/HAM10000_metadata ${data_path}/HAM10000/