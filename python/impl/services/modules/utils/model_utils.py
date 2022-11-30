from impl.services.modules.core.resources import *

def make_model_from_string(model_type : str, target_classes : int) -> nn.Module:
    MODEL_LIST = ["resnet50"]#, "resnet18", "densenet202"]
    MODEL_MAP = {
        "resnet18" : "AutoResnet18",
        "resnet50" : "AutoResnet",
        "densenet202" : "AutoDensenet202"
    }

    if model_type not in MODEL_LIST:
        raise ReiformTrainingException("Model not available")

    model_classname : str = MODEL_MAP[model_type]

    model = eval("{}({})".format(model_classname, target_classes))
    return model

def load_reiform_model(model_path : str, model_type : str, target_classes : int) -> nn.Module:
    
    model : nn.Module = make_model_from_string(model_type, target_classes)

    # Load the state dict
    model = load_pt_model(model, model_path)
    model.eval()
    
    return model

def calculate_batch_size(dims : List[int]):
    prod = 64 * 2 # bytes in long float + buffer
    for v in dims:
        prod*=v

    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    return f // prod