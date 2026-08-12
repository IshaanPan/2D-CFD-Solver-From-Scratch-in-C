# 2D-CFD-Solver-From-Scratch-in-C
Personal Project of Developing a 2D Finite Volume Method Solver to Learn C and CFD Implementation. 
This is still an ongoing project. The base CFD Solver is made, but further work might still be done such as changing to Gauss-Seidl. After this, GPU acceleration will be learned and implemented using CUDA.  

As of August 5, 2026: the mesh logic has been implemented, and the physics logic is being developed. The mesh can be visualised using Matplotlib in Python. 

As of 11 August, 2026: the main physics loops and functions have been implemented. The physics code was almost entirely hand written by me initially, but I was facing divergence issues, so I used Claude to debug.
There were array indexing issues, and the explicit handling of the boundary conditions had some inconsistencies between functions, which resulted in mass not being conserved and the solver diverging. Those issues
were fixed and now the solver converges to residual tolerances lower than 1e-4 on a 100 x 100 grid.

This code uses a non-orthogonal, skewed, body-fitted, structured, collocated grid. 1D flattened arrays are used to store field information, with a column-major indexing approach utilised. The Rhie and Chow interpolation is used to prevent checkerboarding. Physics was applied by basing the equations on the YouTube videos from FluidMechanics101 (namely: https://www.youtube.com/watch?v=yqZ59Xn_aF8&t=1231s and https://www.youtube.com/watch?v=PmEUiUB8ETk&t=1569s). The convective term uses upwinding and the diffusive term uses central differencing, with a deferred correction approach, to ensure correct handling of gradients, while maintaining a strong diagonal matrix to be solved. 


The SIMPLE algorithm is used, whereby per every outer iteration, the x and y 
momentum predictors are solved. Then an inner loop is run to get a converged pressure correction, after which the pressure correction is applied with under-relaxation to get the corrected velocity field and flux. The new arrays are then made into new arrays and the outer loop continues iterating (point Jacobi).

Two wall functions were used, one which was C0 continuous, and the other which is C2 continuous. Additionally, the python scripts can be ran to visualise the mesh and velocity field. The simulation was ran on an HPC, using
20 cores, and finished within ~5 minutes. The top and bottom walls are stationary with no-slip. The inlet is a total pressure inlet condition and the outlet is a static pressure inlet condition.

Things to note:
1) The current solver is not optimised at all. It uses a fixed iteration count to get the pressure correction and this is likely a large bottleneck for instability. Refining the mesh results in NaN errors if the inner loop max iteration is not set high to resolve the pressure correction field well enough. This results in significant increases in computational time, rising quickly to hours as opposed to the quick ~5 minutes for less refined meshes with a max iter for the inner loop of 3000. Even still, if the mesh is too refined, it is likely that the solver diverges to NaNs.
2) Momentum residuals are currently just a measure of the change of the velocity field rather than an actual residual measuring momentum imbalance.
3) Under the current boundary condition (of a Pt-Ps= 50 Pa difference between the inlet and outlet) results in the following velocity field
   <img width="1115" height="581" alt="image" src="https://github.com/user-attachments/assets/4f541274-8934-4e62-a368-cd4be7b4ae20" />

