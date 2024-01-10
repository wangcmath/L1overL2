
%=============================================================
% demo_MRI_brain_T1 ---- Solve a MRI reconstruction problem via L1/L2
%
% Solves
%           min  norm(x,1)/norm(x,2)
%           s.t. Ax = b, p<=x<=q
%
% Reference: "Minimizing L 1 over L 2 norms on the gradient" 
%             Chao Wang, Min Tao, Chen-Nee Chuah, James G Nagy, Yifei Lou 
% Available at: 
%             https://iopscience.iop.org/article/10.1088/1361-6420/ac64fb/
% 
% Author: Chao Wang  
% Date: Jun. 5 2022
%============================================================= 
clear; close all;
% demo code for MRI reconstruction

K = [0.05 0.05];
window = ones(8);
L0 = 1;
filename = 'T1_';
load('brainwebdataset.mat');
%% MRI simulation
F =T1;F = F/max(F(:));
[row,col]= size(F);
N = 217; N0 = sqrt(row*col);
L = 20; 



M0 = double(MRImask_odd(N, L));
Mask0 = M0(:,19:end-18);
Mask = fftshift(Mask0);
data = Mask.*fft2(F)/N0;


list_relerr = [];

pm.Num_iter = 300;
pm.tol = 1e-5;
pm.u_orig = F;
pm.F = F;pm.lb = 0; pm.ub=1;

%% L1/L2
switch L
    case 20
        pm.rho1 = 100;pm.rho2 = 0.01;pm.rho3 = 1;pm.lambda = 0.1;
    case 25
        pm.rho1 = 100;pm.rho2 = .01;pm.rho3 = 0.1;pm.lambda = 0.1;
    case 30
        pm.rho1 = 100;pm.rho2 = .01;pm.rho3 = 0.1;pm.lambda = 0.1;
end
tic;
[u_L1dL2,pm] = mMRrecon_L1dL2_b_s(Mask,data, pm);
timeL1dL2 = toc;RE = norm(abs(u_L1dL2)-F, 'fro')/norm(F, 'fro');
fprintf('L1/L2: Error: %2.2f, runtime: %5.3f; \n',RE,timeL1dL2);


