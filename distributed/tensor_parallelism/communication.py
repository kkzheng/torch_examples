import torch
import torch.distributed as dist
import torch.nn.functional as F


def all2all(tensor,scatter_dim,gather_dim,cur_group,async_op):
    group_size = dist.get_world_size(group=cur_group)
    scatter_tensor_list = list(chunk.contiguous() for chunk in torch.chunk(tensor,chunks=group_size,dim=scatter_dim))
    # if torch.distributed.get_rank() == 0:
    #     print(f'scatter_dim: {scatter_dim} scatter_tensor_list.shape:{[x.shape for x in scatter_tensor_list]}')
    gather_tensor_list = [torch.zeros_like(x) for x in scatter_tensor_list]
    comm = dist.all_to_all(gather_tensor_list, scatter_tensor_list,group = cur_group,async_op = async_op)
    if async_op:
        def wait():
            comm.wait()
            # recieved_tensor = torch.cat(list(torch.chunk(gather_tensor,chunks=group_size,dim=0)),dim=gather_dim)
            recieved_tensor = torch.cat(gather_tensor_list,dim=gather_dim).contiguous()
            # recieved_tensor = recieved_tensor.reshape(expected_shape)
            return recieved_tensor

        return wait()
    recieved_tensor = torch.cat(gather_tensor_list,dim=gather_dim).contiguous()
    # if torch.distributed.get_rank() == 0:
    #     print(f'tensor.shape: {tensor.shape}, group_size: {group_size}, gather_tensor_list.shape:{[x.shape for x in gather_tensor_list]} recieved_tensor.shape:{recieved_tensor.shape}')

    return recieved_tensor


def all_gather(tensor,gather_dim,cur_group,async_op):
    tensor = tensor.contiguous()
    # shape = tensor.shape
    # device = tensor.device
    # cur_group = group_mesh.get_group()
    # group_size = cur_group.size()
    group_size = dist.get_world_size(group=cur_group)
    # gather_shape = list(tensor.shape)
    # gather_shape[gather_dim] = gather_shape[gather_dim] * group_size
    # gather_tensor = torch.empty(gather_shape, dtype=tensor.dtype, 
    #                            device=tensor.device)
    gather_list = [torch.zeros_like(tensor) for _ in range(group_size)]
    # gather_tensor = torch.cat(gather_list,dim=gather_dim)
    # print(f'rank:{dist.get_rank()},tensor.shape:{tensor.shape}')
    comm = dist.all_gather(gather_list,tensor,group=cur_group,async_op=async_op)
    gather_tensor = torch.cat(gather_list,dim=gather_dim)
    # comm = dist.all_gather_into_tensor(gather_tensor,tensor,group=cur_group,async_op=async_op)
    if async_op:
        def wait():
            comm.wait()
            gather_tensor = torch.cat(gather_list,dim=gather_dim)
            return gather_tensor

        return wait()
    # gather_tensor = torch.cat(gather_list,dim=gather_dim)

    return gather_tensor


'''
forward:
    1) 保存上下文供 backward 使用
    2) 执行 all2all 操作,将 tensor 按照 scatter_dim 进行切分，然后按照 gather_dim 进行拼接
backward:
    1) 执行 all2all 操作,将 grad_outputs 按照 gather_dim 进行切分，然后按照 scatter_dim 进行拼接
        注意这里 scatter_dim 和 gather_dim 是相反的
    2) backward 返回的 tuple, 必须和 forward 的输入一一对应
'''
class _All2All(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor,
        scatter_dim,
        gather_dim,
        cur_group,
        async_op
    ):
        ctx.cur_group = cur_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.async_op = async_op
        return all2all(tensor=tensor,scatter_dim=scatter_dim,gather_dim=gather_dim,cur_group=cur_group,async_op=async_op)
    
    @staticmethod
    def backward(ctx, grad_outputs):
        rank = dist.get_rank()
        # print(f'all2all rank:{rank};grad_outputs.type:{type(grad_outputs)};grad_outputs.shape:{grad_outputs.shape}')
        input_t = grad_outputs
        return (all2all(input_t,ctx.gather_dim,ctx.scatter_dim,ctx.cur_group,False),None,None,None,None)

class _Allgather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        tensor,
        gather_dim,
        cur_group,
        async_op
    ):
        ctx.gather_dim = gather_dim
        # ctx.group_mesh = group_mesh
        ctx.cur_group=cur_group
        ctx.async_op = async_op
        # ctx.split_dim = tensor.shape[gather_dim]//group_mesh.get_group().size()
        return all_gather(tensor=tensor,gather_dim=gather_dim,cur_group=cur_group,async_op= async_op)
    
    @staticmethod
    def backward(ctx,grad_outputs):
        rank = dist.get_rank()
        # print(f'allgather rank:{rank};grad_outputs.type:{type(grad_outputs)};grad_outputs.shape:{grad_outputs.shape};grad_outputs[0]:{grad_outputs[0]}')
        # sp_group = ctx.group_mesh.get_group()
        sp_group=ctx.cur_group
        # sp_group_size = sp_group.size()
        sp_group_size = dist.get_world_size(group=sp_group)
        rank_in_group = dist.get_group_rank(group=sp_group,global_rank=rank)
        # print(f'allgather rank:{rank};grad.shape:{grad_outputs.split(grad_outputs.shape[ctx.gather_dim]//sp_group_size,dim=ctx.gather_dim)[rank_in_group].shape}')
        return (grad_outputs.split(grad_outputs.shape[ctx.gather_dim]//sp_group_size,dim=ctx.gather_dim)[rank_in_group],None,None,None)


class _Slice(torch.autograd.Function):
    @staticmethod
    def forward(ctx,tensor,slice_dim,group_mesh):
        # ctx.device = tensor.device
        seq_len = tensor.shape[1]
        local_sp = seq_len // group_mesh.size()
        rank = dist.get_rank()
        sp_group = group_mesh.get_group()
        group_rank = dist.get_group_rank(sp_group,rank)
        device = torch.device("cuda",rank)
        # tensor_local = tensor[:,group_rank*local_sp:(group_rank+1)*local_sp,:,:]
        tensor_local = tensor.split(local_sp,dim=slice_dim)[group_rank]
        tensor_local = tensor_local.to(device)
        ctx.sp_group = sp_group
        ctx.slice_dim = slice_dim
        return tensor_local
    @staticmethod
    def backward(ctx,grad_outputs):
        rank = dist.get_rank()
        # grad_outputs = grad_outputs.to(ctx.device)
        # print(f'rank:{rank},grad_outputs.device:{grad_outputs.device};grad_outputs.shape:{grad_outputs.shape}')
        grad_return = all_gather(tensor=grad_outputs,gather_dim=ctx.slice_dim,cur_group=ctx.sp_group,async_op= False).to("cpu")
        return (grad_return,None,None)


def minimal_pad_to_divisible(tensor: torch.Tensor, sp_size: int, dim: int = 1, pad_value: float = 0.0):
    """
    对三维或更高维度的tensor在指定维度进行最小化padding，使其长度能被 sp_size 整除。

    Args:
        tensor: 输入的PyTorch tensor (例如：[B, L, C] 或 [B, H, W, C] 等)。
        sp_size: 要求的最小分割尺寸。
        dim: 需要进行padding的维度索引（默认为 1，即第二维）。
        pad_value: 填充的值（默认为 0.0）。

    Returns:
        padded_tensor: 填充后的 tensor。
    """
    
    current_size = tensor.size(dim)
    
    padding_len = (sp_size - current_size % sp_size) % sp_size
    
    if padding_len == 0:

        return tensor, 0 

    padding_dims = [0] * (2 * tensor.dim())
    
    pad_index = 2 * (tensor.dim() - dim - 1) + 1
    
    if pad_index < len(padding_dims):
        padding_dims[pad_index] = padding_len
    else:
        raise ValueError("Invalid dimension index.")

    pad = tuple(padding_dims)

    padded_tensor = F.pad(tensor, pad=pad, mode='constant', value=pad_value)
    
    return padded_tensor, padding_len