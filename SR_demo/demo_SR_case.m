
%=============================================================
% demo_SR_case ---- Solve a super-resolution problem via L1/L2
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
K = [0.05 0.05];
window = ones(8);
L0 = 1;
filename = 'SR_';
I = imread('phpDlNSHI.png');
%% MRI simulation
N = 344; F = double(I(1:2:end,1:2:end));
% N = 688; F = double(I);
F = F/max(F(:));
slen = 70;
Mask = zeros(N); Mask(N/2-slen/2+1:N/2+slen/2,N/2-slen/2+1:N/2+slen/2)=1;
Mask = fftshift(double(Mask));
data = Mask.*fft2(F)/N;



pm.Num_iter = 300;
pm.tol = 1e-5;
pm.F=F;pm.lb = 0; pm.ub=1;



%% L1/L2
switch slen
    case 50
        pm.rho1 = 10;pm.rho2 = 1;pm.rho3 = 10;pm.lambda = 1;
    case 60
        pm.rho1 = 10;pm.rho2 = 1;pm.rho3 = 10;pm.lambda = 1;
    case 70
        pm.rho1 = 1;pm.rho2 = 10;pm.rho3 = 100;pm.lambda = 10;
end
tic;
[u_L1dL2,pm] = mMRrecon_L1dL2_b_s(Mask,data, pm);
timeL1dL2 = toc;RE = norm(abs(u_L1dL2)-F, 'fro')/norm(F, 'fro');
fprintf('L1/L2: Error: %2.2f, runtime: %5.3f; \n',RE,timeL1dL2);
