function [u,pm] = mMRrecon_L1dL2_b_s(R, f, pm)
%=============================================================
% ADMM for the L1/L2 constrained model for MRI/SR reconstruction
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

[rows,cols] = size(f);

Num_iter = 3000; rho1 = 50; rho2 = 50;Imaxit = 10; 
u0 = zeros(rows,cols); lambda = 50; rho3 = 1;
tol = 1e-5;
lb = 0; ub = 1;
u_orig = pm.F;
if isfield(pm,'lb'); lb = pm.lb; end
if isfield(pm,'ub'); ub = pm.ub; end
if isfield(pm,'rho1'); rho1 = pm.rho1; end
if isfield(pm,'rho2'); rho2 = pm.rho2; end
if isfield(pm,'rho3'); rho3 = pm.rho3; end
if isfield(pm,'tol'); tol = pm.tol; end
if isfield(pm,'u0'); u0 = pm.u0; end
if isfield(pm,'lambda'); lambda = pm.lambda; end
if isfield(pm,'Num_iter'); Num_iter = pm.Num_iter; end


u = u0; v = u;


% Build Kernels
scale = sqrt(rows*cols);


uker = zeros(rows,cols);
uker(1,1) = 4;uker(1,2)=-1;uker(2,1)=-1;uker(rows,1)=-1;uker(1,cols)=-1;
uker = lambda*(conj(R).*R)+(rho1+rho2)*fft2(uker) + rho3*ones(rows,cols);
% uker(uker<0)=0; uker(uker>1) =1;

ff = f;
f0 = ff; f = f0; murf = ifft2(lambda*R.*f0)*scale;

% Augmented Lagrangian Parameters
dx = zeros(rows,cols);
dy = zeros(rows,cols);
hx = zeros(rows,cols);
hy = zeros(rows,cols);

bx = zeros(rows,cols);
by = zeros(rows,cols);
cx = zeros(rows,cols);
cy = zeros(rows,cols);
h = bx;
step = 0;
list_fo = zeros(Num_iter*Imaxit,1);
list_fc = zeros(Num_iter*Imaxit,1);
list_fcpu = list_fc;
tstart = tic;
list_re = zeros(Num_iter,1);list_e = list_re;
list_o = list_re;
list_c = list_re;
list_cpu = list_re;

List = zeros(Num_iter,2);
for j = 1:Num_iter
        uold = u;
        for ii = 1:Imaxit
            step = step + 1; 
        % u-update
        rhs = murf+rho1*Dxt(dx-bx)+rho1*Dyt(dy-by)+rho2*Dxt(hx-cx)+...
            rho2*Dyt(hy-cy)+ rho3*(v-h);
        u = real(ifft2(fft2(rhs)./uker));
        v = u+h;
        v(v<lb)=lb; v(v>ub) =ub;
        h  = h + u - v;
%         u(u<0)=0; u(u>1) =1;
        % dx,dy-update
        Dxu = Dx(u);
        Dyu = Dy(u);
        hnorm = sqrt(norm(hx(:))^2+norm(hy(:))^2);
        dx = shrink(Dxu+bx, 1/(rho1*hnorm));
        dy = shrink(Dyu+by, 1/(rho1*hnorm));
        
        bx = bx+(Dxu-dx);
        by = by+(Dyu-dy);
        
        f = f+f0-R.*fft2(u)/scale;
        murf = ifft2(lambda*R.*f)*scale;
        list_fcpu(step) = toc(tstart);
        list_fo(step) =  (norm(Dxu(:),1)+norm(Dyu(:),1))/sqrt(norm(Dxu,'fro').^2+norm(Dyu,'fro').^2);
        list_fc(step) = norm(u-u_orig,'fro')/norm(u_orig,'fro');
        
        end

        % hx,hy-update
        d1 = Dxu + cx;
        d2 = Dyu + cy;
        etha = sqrt(norm(d1(:))^2+norm(d2(:))^2);
        c = norm(dx(:),1)+norm(dy(:),1);
        [hx, hy,tau] = mupdate_h(c,etha,rho2,d1,d2);

        % bx,by,cx,cy-update
        
        
        cy = cy+(Dyu-hy);
        cx = cx+(Dxu-hx);
        list_o(j) = list_fo(step);
        list_c(j) = list_fc(step);
        list_cpu(j) = list_fcpu(step);
        
        
        List(j,1) = (norm(Dxu(:),1)+norm(Dyu(:),1))/sqrt(norm(Dxu(:)).^2+norm(Dyu(:)).^2);
        List(j,2) = norm(uold-u,'fro')/norm(uold,'fro');

end

list_o(j+1:end) = [];
list_c(j+1:end) = [];
list_cpu(j+1:end)=[];
pm.obj = list_o;
pm.fobj = list_fo;
pm.rmse = list_c;
pm.frmse = list_fc;
pm.cpu = list_cpu;
pm.fcpu = list_fcpu;
pm.i = j;
pm.L = List;

end

function d = Dx(u)
[rows,cols] = size(u);
d = zeros(rows,cols);
d(:,2:cols) = u(:,2:cols)-u(:,1:cols-1);
d(:,1) = u(:,1)-u(:,cols);
end

function d = Dxt(u)
[rows,cols] = size(u);
d = zeros(rows,cols);
d(:,1:cols-1) = u(:,1:cols-1)-u(:,2:cols);
d(:,cols) = u(:,cols)-u(:,1);
end

function d = Dy(u)
[rows,cols] = size(u);
d = zeros(rows,cols);
d(2:rows,:) = u(2:rows,:)-u(1:rows-1,:);
d(1,:) = u(1,:)-u(rows,:);
end

function d = Dyt(u)
[rows,cols] = size(u);
d = zeros(rows,cols);
d(1:rows-1,:) = u(1:rows-1,:)-u(2:rows,:);
d(rows,:) = u(rows,:)-u(1,:);
end


function [tau] = mfindroot(c,eta,rho)

if c == 0 || eta == 0 
    tau = 1;
else
    a = 27*c/(rho*(eta^3)) + 2;
    C = ((a + (a^2 - 4)^0.5)/2)^(1/3);
    tau = (1 + C + 1/C)/3;
end

end


function [hx, hy,tau] = mupdate_h(c,etha,rho,d1,d2)

if etha == 0 
   hx = (c/rho)^(1/3)*ones(size(d1))/sqrt(numel(d1)*2);
   hy = (c/rho)^(1/3)*ones(size(d2))/sqrt(numel(d2)*2);
else
    a = 27*c/(rho*(etha^3)) + 2;
    C = ((a + (a^2 - 4)^0.5)/2)^(1/3);
    tau = (1 + C + 1/C)/3;
    hx = tau*d1;
    hy = tau*d2;
end

end
function z = shrink(x,r)
z = max(0,x - r) - max(0,-x - r);
end

