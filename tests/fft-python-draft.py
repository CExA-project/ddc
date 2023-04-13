import numpy as np;
import sympy as sp;

x, k = sp.symbols('x k', real=True);
f = sp.exp(-x**2/2);
F = 1/sp.sqrt(2*sp.pi)*sp.fourier_transform(f, x, k);
f_np = sp.lambdify(x, f, 'numpy');
F_np = sp.lambdify(k, F, 'numpy');
print(F);

Nx = 32;
a = -2*np.pi;
b = 2*np.pi;

lambda_N = (b-a)/Nx;
f_s = 1/lambda_N/2;
k_s = 2*np.pi*f_s; # pi*Nx/(b-a)
print(k_s);
# Solve equation : 2*k_s = b-a => 2*pi*(Nx-1)/(b-a) = b-a => Nx = (b-a)^2/2/pi
Nx_opti = (b-a)**2/2/np.pi;
print(Nx_opti);

x_mesh = [a + (b-a)/Nx*(1/2 + i) for i in range(0,Nx)];
print("----- One dimension -----");
f= [f_np(x) for x in x_mesh];
print(f);
F = (b-a)/Nx/np.sqrt(2*np.pi)*np.abs(np.fft.fftn(f));
print(F);
print([F_np(k/2/np.pi) for k in [2*np.pi/(b-a)*i for i in range(0,int(Nx/2)+1)]]);
print(F[0:int(Nx/2+1)]-[F_np(k/2/np.pi) for k in [2*np.pi/(b-a)*i for i in range(0,int(Nx/2)+1)]]);
