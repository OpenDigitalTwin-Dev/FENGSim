import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftfreq

def fourier_poisson_solver():
    # 2d poisson equation
    N = 64                    
    L = 2 * np.pi            
    x = np.linspace(0, L, N, endpoint=False)  
    y = x.copy()
    X, Y = np.meshgrid(x, y)
    
    # u(x,y) = sin(2x)cos(3y)
    u_exact = np.sin(2*X) * np.cos(3*Y)
    f = -13 * np.sin(2*X) * np.cos(3*Y)
    
    u_numerical = solve_poisson_spectral(f, L)

    error = np.max(np.abs(u_numerical - u_exact))
    print(f"error: {error:.2e}")
    
    # visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im1 = axes[0].imshow(f, extent=[0, L, 0, L], origin='lower')
    axes[0].set_title('source f(x,y)')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(u_exact, extent=[0, L, 0, L], origin='lower')
    axes[1].set_title('analytical solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    plt.colorbar(im2, ax=axes[1])
    
    im3 = axes[2].imshow(u_numerical, extent=[0, L, 0, L], origin='lower')
    axes[2].set_title('numerical solution')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.show()
    
    return
    plt.figure(figsize=(8, 6))
    error_contour = plt.contourf(X, Y, np.abs(u_numerical - u_exact), 20)
    plt.colorbar(error_contour)
    plt.title('error')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.axis('image')
    plt.show()

def solve_poisson_spectral(f, L):    
    N, M = f.shape

    kx = 2 * np.pi / L * np.fft.fftfreq(N, 1/N)
    ky = 2 * np.pi / L * np.fft.fftfreq(M, 1/M)
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    K2 = KX**2 + KY**2
    
    K2[0, 0] = 1.0
    
    f_hat = fft2(f)
    
    u_hat = -f_hat / K2
    
    u_hat[0, 0] = 0
    
    u = np.real(ifft2(u_hat))
    
    return u

if __name__ == "__main__":
    fourier_poisson_solver()
