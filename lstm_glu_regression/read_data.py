import json

with open("/data/DataSetFullSilence.json","r") as f:

# with open("D:\Listener\DataDriven\data\DataProcessingCode\DataSetFullSilence.json","r") as f:
    data_dict = json.load(f)

for key in data_dict:
    print(key)

blendshape = data_dict["sp_blendshape"][0]
print(len(blendshape[0]))

with open("/data/DataSetFullSilenceWordDict.json","w",encoding="utf-8") as f:
    json.dump(data_dict["word_dict"], f)