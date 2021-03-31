import torch

def sort_tensor(inp_tensor, sort_column_id):
    """
    Sort a tensor according the contents of a column
    Params ::
    inp_tensor: Tensor: Tensor to be sorted
    sort_column_id: int: index of column used for sorting
    Return ::
    (out_tensor, idx): Tuple: (Sorted Tensor, idx used for sorting)
    """
    sort_column = inp_tensor[:, sort_column_id] 
    _, idx = sort_column.sort()
    out_tensor = inp_tensor.index_select(0, idx)
    return (out_tensor, idx)

def tensor_pop(inp_tensor, to_pop, is_index=True):
    """
    Pop elements from an input tensor
    Params ::
    inp_tensor: tensor: Input Tensor
    to_pop: array array like: 
        collection of indexes or elements to pop from inp_tensor
    is_index: Boolean: if set to True, to_pop is treated as indices. If False
        to_pop is treated as list of elements. Default is True
    Return ::
    Tuple(out_tensor, popped_elements):
        out_tensor: Tensor of type inp_tensor: Input tensor with the 
            popped elements removed
        popped_elements: Tensor of type inp_tensor: Tensor of popped rows
    """
    if is_index is True:
        idx_to_keep = torch.tensor([id for id in range(inp_tensor.size(0)) \
            if id not in to_pop], device=device, dtype=torch.long)
        to_pop = torch.tensor(to_pop, device=device, dtype=torch.long)
        popped_elements = inp_tensor.index_select(0, to_pop)
        out_tensor = inp_tensor.index_select(0, idx_to_keep)

    else:
        raise NotImplementedError()
    return (out_tensor, popped_elements)