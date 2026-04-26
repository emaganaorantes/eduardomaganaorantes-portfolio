

import torch
import numpy as np
from torch import nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

plt.close('all')

#%matplotlib qt

######################################################

#define neural network
class Network(nn.Module):
    def __init__( self , n ):
        super().__init__()
    
  
        #two hidden layers (weights and biases for linear transform)
        self.layer_1 = nn.Linear( 2 , n )
        self.layer_2 = nn.Linear( n , n )
        
        #output layer is a dot product
        self.output = nn.Linear( n , 1 , bias=False )
        
        #activation function
        self.act = nn.Tanh()
      
    def forward( self , x):
        
        #two hidden layer feed forward neural network
        y = self.layer_1(x)
        y = self.act(y)
        y = self.layer_2(y)
        y = self.act(y)
     
        #enforce zero displacement BC on the two boundaries at x=0 and x=1
        #boudary conditions are enforced in the NN equivalent of "basis functions"
        #y = torch.sin( np.pi * x ) * y
        
        #dot product to go from second hidden layer to solution
        y = self.output(y)
        
        #dealing with zero  boundary condition
        x1 = x[:,0].reshape(-1,1)
        x2 = x[:,1].reshape(-1,1)
        
        g = torch.sin( np.pi * x1 ) * torch.sin( np.pi * x2 )
        
        y = y * g
        
        return y
    
    #integral of squared error of differential equation (pass in the integration grid)
    def error( self , x ):
        
        #evaluate source term
        F = f(x)
        
        #material constant?
        K = k(x)
        
        #enables pytroch to compute the gradient of the solution with respect to this input
        x.requires_grad = True
        
        #evaluate the solution
        u = self.forward(x)
        
        #compute first derivative of the solution at each integration point
        u_grad = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        u_x,u_y = u_grad[:,0].reshape(-1,1), u_grad[:,1].reshape(-1,1)
        
        #Material scaled part
        A_x,A_y = K * u_x, K * u_y 
        
        #compute second derivative of the solution at each integration point
        grad_Ax = torch.autograd.grad(A_x , x , grad_outputs=torch.ones_like(A_x) , create_graph=True )[0]
        
        grad_Ay = torch.autograd.grad(A_y , x , grad_outputs=torch.ones_like(A_y) , create_graph=True )[0]
        
        #laplace = (grad_ux[:,0] + grad_uy[:,1]).reshape(-1,1)
        
        #divergence 
        divergence = (grad_Ax[:,0] + grad_Ay[:,1]).reshape(-1,1)
        
        #integral of squared error
        L = 0.5 * dA * torch.sum( ( divergence + F )**2  )
        
        return L
        
    def error_insulated( self , x , xb ):
        
        #evaluate source term
        F = f(x)
        
        #material constant?
        K = k(x)
        
        #enables pytroch to compute the gradient of the solution with respect to this input
        x.requires_grad = True
        
        #evaluate the solution
        u = self.forward(x)
        
        #compute first derivative of the solution at each integration point
        u_grad = torch.autograd.grad( u , x , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        u_x,u_y = u_grad[:,0].reshape(-1,1), u_grad[:,1].reshape(-1,1)
        
        #Material scaled part
        A_x,A_y = K * u_x, K * u_y 
        
        #compute second derivative of the solution at each integration point
        grad_Ax = torch.autograd.grad(A_x , x , grad_outputs=torch.ones_like(A_x) , create_graph=True )[0]
        
        grad_Ay = torch.autograd.grad(A_y , x , grad_outputs=torch.ones_like(A_y) , create_graph=True )[0]
        
        #laplace = (grad_ux[:,0] + grad_uy[:,1]).reshape(-1,1)
        
        #divergence 
        divergence = (grad_Ax[:,0] + grad_Ay[:,1]).reshape(-1,1)
        
        xb.requires_grad = True
        
        u = self.forward(xb)
        
        u_grad = torch.autograd.grad( u , xb , grad_outputs=torch.ones_like(u) , create_graph=True )[0]
        
        norm = torch.zeros((pts-1, 2))
        norm[:,0] = torch.ones(pts-1)
        
        u_grad_n = torch.einsum( 'ij,ij->i' , u_grad , norm )
        
        #integral of squared error
        L = 0.5 * dA * torch.sum( ( divergence + F )**2  )
        
        return L
        
    
######################################################

#source term
def f(x):
    x1 = x[:,0].reshape(-1,1)
    x2 = x[:,1].reshape(-1,1)
    
    vals = 100 * torch.sin(2*np.pi*x1) * torch.sin(2*np.pi*x2)
    
    return vals

def u_guess(x):
    x1 = x[:,0].reshape(-1,1)
    x2 = x[:,1].reshape(-1,1)
    
    vals = torch.sin(np.pi * x1) * torch.sin(np.pi * x2)
    
    return vals 
 
#material constant
def k(x):
     x1 = x[:,0].reshape(-1,1)
     x2 = x[:,1].reshape(-1,1)
     
     vals = 1 + x1 + x2
    
     return vals

#integration grid, x = [0,1] domain
pts = 35
x = torch.linspace(0,1,pts)
dx = x[1]-x[0]
dA = dx**2
x += dx/2
x = x[:-1]

grid = []
for i in range(pts-1):
    for j in range(pts-1):
        grid +=[ [ x[i] , x[j] ] ]

X = torch.tensor( grid , dtype=torch.float32 )

boundary_grid = torch.zeros((pts-1,2))
boundary_grid[:,0] = torch.ones(pts-1)
boundary_grid[:,1] = x

# #show integration grid
# plt.figure()
# plt.scatter( X[:,0].detach() , X[:,1].detach() )
# plt.title('Integration grid')


#width of two hidden layers in the network
n = 20

#initialize network
u = Network(n)

u.error_insulated(X, boundary_grid)

total_params = sum(p.numel() for p in u.parameters())
print(f"Total parameters: {total_params}")

#number of steps in gradient descent
epochs = 500

#size of gradient descent step
lr = 5e-3

#initialize "ADAM" optimization (one of the variants)
optimizer = torch.optim.Adam( u.parameters() , lr=lr )

#store values of objective at each step
losses = np.zeros(epochs)

#training loop
for i in range(epochs):
    
    #zero the gradients (they accumulate otherwise)
    optimizer.zero_grad()
    
    #compute the objective value
    loss = u.error(X)
    
    #have pytorch compute gradients wrt NN parameters
    loss.backward()
    
    #use gradient to update the parameters according to ADAM optimizer
    optimizer.step()
    
    #store loss (objective) value
    losses[i] = round( loss.item() , 4 )
    
    if i % 500 == 0:
        print(f'Epoch {i}, Loss {losses[i]}')
        
        # Compute NN solution
u_nn = u.forward(X).detach()

# Compute guessed solution
u_true = u_guess(X)

# Compute absolute error
error = torch.abs(u_nn - u_true)

# Plot error
plt.figure()
X1, X2 = X[:, 0].detach().numpy().reshape(pts-1, pts-1), X[:, 1].detach().numpy().reshape(pts-1, pts-1)
error_reshaped = error.detach().numpy().reshape(pts-1, pts-1)

plt.contourf(X1, X2, error_reshaped, levels=20, cmap="coolwarm")
plt.colorbar(label="Absolute Error")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Contour Plot of Error")
plt.show()

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X1, X2, error_reshaped, cmap="coolwarm")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("Error Magnitude")
ax.set_title("3D Surface Plot of Error")

plt.show()




#plot the error as a function of step in optimization, error should converge to zero
plt.figure()
plt.plot( np.log( losses ) )
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training')
plt.show()


#for plotting solution
X1 , X2 = np.meshgrid( x , x )
Z = u.forward(X).detach()
Z = torch.reshape( Z , (pts-1,pts-1) )


#surface plot
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface( X1 , X2 , Z , cmap='viridis') 








