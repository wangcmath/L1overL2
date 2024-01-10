function u = MRreconL1_b(R,f, pm)
%function [u, err, cpu] = MRreconTV(R,f, mu, rho,  nIter, ifcon, u_orig)

% |x|+|y| + 0.5*rho*||Dx u-x+bx||^2 + 0.5*rho*||Dy u-y+by||^2
% s.t. RFu = f
%
% using the Split Bregman method ftp://ftp.math.ucla.edu/pub/camreport/cam08-29.pdf
%


[rows,cols] = size(f);

lambda = 20; rho = 1; 
u0 = zeros(rows,cols); tol = 1e-5; Num_iter = 3000;

if isfield(pm,'lb'); lb = pm.lb; end
if isfield(pm,'ub'); ub = pm.ub; end
if isfield(pm,'rho'); rho = pm.rho; end
if isfield(pm,'lambda'); lambda = pm.lambda; end
if isfield(pm,'rho3'); rho3 = pm.rho3; end
if isfield(pm,'Num_iter'); Num_iter = pm.Num_iter; end
if isfield(pm,'u_orig'); u_orig = pm.u_orig; end
if isfield(pm,'u0'); u0 = pm.u0; end
if isfield(pm,'tol'); tol = pm.tol; end; % inner iteration tolerance

[rows,cols] = size(f);

% Reserve memory for the auxillary variables
f0 = f;
u = u0; v = u;
x = zeros(rows,cols);
y = zeros(rows,cols);
bx = zeros(rows,cols);
by = zeros(rows,cols);
h = bx;
% Build Kernels
scale = sqrt(rows*cols);
murf = ifft2(lambda*(conj(R).*f))*scale;

uker = zeros(rows,cols);
uker(1,1) = 4;uker(1,2)=-1;uker(2,1)=-1;uker(rows,1)=-1;uker(1,cols)=-1;
uker = lambda*(conj(R).*R)+rho*fft2(uker) + rho3*ones(rows,cols);
List = zeros(Num_iter,4);

%  Do the reconstruction
for j = 1:Num_iter
    uold = u;
   % for inner = 1:Num_iter/50       
        % update u
        rhs = murf+rho*Dxt(x-bx)+rho*Dyt(y-by) + rho3*(v-h);
        u = ifft2(fft2(rhs)./uker);
%          u(u<0)=0; u(u>1) =1;
        v = u+h;
        v(v<lb)=lb; v(v>ub) =ub;
        % update x and y
        dx = Dx(u);
        dy  =Dy(u);
        
        % anisotropic TV
        x = shrink(dx+bx, 1/rho);
        y = shrink(dy+by, 1/rho);
        
        % update bregman parameters
        bx = bx+dx-x;
        by = by+dy-y;
        h  = h + u - v;
   % end
    
    f = f+f0-R.*fft2(u)/scale;
    murf = ifft2(lambda*R.*f)*scale;
    
    
%         List(j,1) = norm(abs(u)-pm.F, 'fro')/norm(pm.F, 'fro');
%         List(j,2) = (norm(dx(:),1)+norm(dy(:),1));
%         List(j,3) = sqrt(norm(dx(:)-x(:))^2+norm(dy(:)-y(:))^2);
%         List(j,4) = sqrt(norm(dx(:)-x(:))^2+norm(dy(:)-y(:))^2);
        List(j,5) = norm(uold-u,'fro')/norm(uold,'fro');
        if List(j,5) < tol
            break;
        end
    
end


return;


function d = Dx(u)
[rows,cols] = size(u);
d = zeros(rows,cols);
d(:,2:cols) = u(:,2:cols)-u(:,1:cols-1);
d(:,1) = u(:,1)-u(:,cols);
return

function d = Dxt(u)
[rows,cols] = size(u);
d = zeros(rows,cols);
d(:,1:cols-1) = u(:,1:cols-1)-u(:,2:cols);
d(:,cols) = u(:,cols)-u(:,1);
return

function d = Dy(u)
[rows,cols] = size(u);
d = zeros(rows,cols);
d(2:rows,:) = u(2:rows,:)-u(1:rows-1,:);
d(1,:) = u(1,:)-u(rows,:);
return

function d = Dyt(u)
[rows,cols] = size(u);
d = zeros(rows,cols);
d(1:rows-1,:) = u(1:rows-1,:)-u(2:rows,:);
d(rows,:) = u(rows,:)-u(1,:);
return


function z = shrink(x,r)
z = sign(x).*max(abs(x)-r,0);
return;

