function output=Jacobian_transpose(input)
%INPUT is (theta1, theta2, d3, theta4,xd_dot_plus_Ke) 
theta1=input(5,:); d1=0; a1=0.5; alpha1=0; %parameter link1
theta2=input(6,:); d2=0; a2=0.5; alpha2=0; %parameter link2
theta3=0; d3=input(7,:); a3=0; alpha3=0;  %parameter link3
theta4=input(8,:); d4=0; a4=0; alpha4=0;  %parameter link4
 T01=DHmatrix(theta1,d1,a1,alpha1);
 T12=DHmatrix(theta2,d2,a2,alpha2);
 T23=DHmatrix(theta3,d3,a3,alpha3);
 T34=DHmatrix(theta4,d4,a4,alpha4);
 T02=T01*T12;
 T03=T01*T12*T23;
 T04=T01*T12*T23*T34;
P0=[0;0;0];
P1=T01(1:3,4);
P3=T03(1:3,4);
P4=T04(1:3,4);

Z0=[0;0;1];
Jv1=cross(Z0,(P4-P0));
Jv2=cross(Z0,(P4-P1));
Jv3=Z0;
Jv4=cross(Z0,(P4-P3));
Jv=[Jv1 Jv2 Jv3 Jv4;1 1 0 1];
output=Jv'*input(1:4,:);
end

function [Mdh] = DHmatrix(theta,d,a,alpha)
%DHMATRIX Summary of this function goes here
%   inputannya DH parameter
Mdh=[cosd(theta) -sind(theta)*cosd(alpha) sind(theta)*sind(alpha) a*cosd(theta);
     sind(theta) cosd(theta)*cosd(alpha)  -cosd(theta)*sind(alpha) a*sind(theta);
     0,sind(alpha),cosd(alpha),d;
     0,0,0,1];
end