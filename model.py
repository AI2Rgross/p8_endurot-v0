import torch
import torch.nn as nn
import torch.nn.functional as F
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size,seed):
        super(QNetwork, self).__init__()
        w=state_size[0]
        h=state_size[1]
        #print("w",w)
        #print("h",h)
       
        self.seed = torch.manual_seed(seed)

        self.conv1 = nn.Conv2d(1, 16,  kernel_size=5, stride=2)
        #self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=2, stride=2)
        #self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, stride=2)
        #self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            #print("test",size,kernel_size,(kernel_size - 1),(size - (kernel_size - 1) - 1) // stride,(size - (kernel_size - 1) - 1) // stride  + 1)
            return (size - (kernel_size - 1) - 1) // stride  + 1
        
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w,5,2),3,2),2,2)
        #print("convw",conv2d_size_out(w,5,2), conv2d_size_out(conv2d_size_out(w,5,2),3,2), conv2d_size_out(conv2d_size_out(conv2d_size_out(w,5,2),3,2),2,2))
       #print("convh",conv2d_size_out(h,5,2), conv2d_size_out(conv2d_size_out(h,5,2),3,2), conv2d_size_out(conv2d_size_out(conv2d_size_out(h,5,2),3,2),2,2))

        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h,5,2),3,2),2,2)
        linear_input_size = convw * convh * 32
        #print("linear_input_size",linear_input_size)
        self.head = nn.Linear(linear_input_size, action_size) # 448 or 512
 
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu( self.conv1(x))
        x = F.relu( self.conv2(x))
        x = F.relu( self.conv3(x))
        flat =x.view(x.size(0), -1)
        #print(len(flat),len(flat[0]))
        return self.head(flat)
    
    
