from .iresnet import iresnet50

def get_model(**kwargs):
  return iresnet50(False, **kwargs)

 