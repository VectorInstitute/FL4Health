data_path="fl4health/utils/datasets"

unzip ${data_path}/release_v0.zip -d ${data_path}/
mv ${data_path}/release_v0 ${data_path}/Derm7pt
rm -rf ${data_path}/release_v0