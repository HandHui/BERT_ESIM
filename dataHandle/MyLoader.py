from torch.utils.data import Dataset

### Rawdata --> dataser
### __init__()/__getitem__/__len__() 
class Mydata(Dataset):
    def __init__(self,data):
        self.data = data
    def __getitem__(self,idx):
        assert idx<len(self.data)
        return self.data[idx]
    def __len__(self):
        return len(self.data)


