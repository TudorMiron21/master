 % 2 rooms
 A_2=[52 61; 39 52; 51 78; 52 88; 58 59; 50 60; 80 125; 56 62; 50 60; 55 63; 46 66; 54 59; 58 65; 55 63;...
     64 115; 41 50; 59 78; 53 100; 75 85; 52 61; 54 61; 52 65; 51 57; 50 62; 49 65; 40 55; 54 95; 45 52; 54 83;...
     37 51; 67 119; 65 90; 40 52; 57 82; 55 60; 45 60];
%  figure
%  plot(A_2(:,1),A_2(:,2),'o','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',3)
 
 %%
 % 1 room
 A_1=[31 45; 30 44; 32 45; 35 56; 34 51; 43 52; 31 49; 33 50; 34 40; 40 65; 15 23; 31 50; 41 50];
 %%
 % 3 rooms
 A_3=[90 122; 67 83; 67 75; 90 169; 73 109; 70 92; 66 82; 77 85; 73 74; 63 73; 76 115; 70 110; 73 86; 68 85];
 %%
 %4 rooms
 A_4=[78 100; 83 130; 78 115; 90 95; 86 100; 96 125; 104 230; 105 135; 90 130; 79 120; 90 107 ;95 107];
 
 %% linear regression
 A=[A_1;A_2;A_3;A_4];
 figure
 plot(A(:,1),A(:,2),'o','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',3)
 title('Area vs House price')
 %% multiple linear regression
 X1=[2*ones(size(A_2,1),1) A_2(:,1);ones(size(A_1,1),1) A_1(:,1);3*ones(size(A_3,1),1) A_3(:,1);4*ones(size(A_4,1),1) A_4(:,1)];
 X=[ones(size(X1,1),1) X1];
 Y=[A_2(:,2);A_1(:,2);A_3(:,2);A_4(:,2)];
 a=(X'*X)^(-1)*X'*Y;
 figure
 plot3(X1(:,1),X1(:,2),Y,'o','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',6)
 hold on
 x=1:4;
 y=linspace(min(X1(:,2)),max(X1(:,2)),2);
 [XX,YY]=meshgrid(x,y);
 ZZ=a(1)+a(2)*XX+a(3)*YY;
 surf(XX,YY,ZZ)
 plot3(2,49,a(2)*2+a(3)*49+a(1),'o','MarkerEdgeColor','k','MarkerFaceColor','m','MarkerSize',10)
 plot3(3,86,a(2)*3+a(3)*86+a(1),'o','MarkerEdgeColor','k','MarkerFaceColor','m','MarkerSize',10)
 plot3(4,130,a(2)*4+a(3)*130+a(1),'o','MarkerEdgeColor','k','MarkerFaceColor','m','MarkerSize',10)
 %% Cazul apartamentelor cu 2 camere
  A_2=[52 61; 39 52; 51 68; 52 52; 58 59; 50 60; 45 60;  56 62; 50 60; 55 63; 46 66; 54 59; 58 65; 55 63];
     
  %64 115; 41 50; 59 78; 53 100; 75 85; 52 61; 54 61; 52 65; 51 57; 50 62; 49 65; 40 55; 54 95; 45 52; 54 83;...
   
  %37 51; 67 119; 65 90; 40 52; 57 82; 55 60; 80 125];
  n=12;
 Er=zeros(1,n);
 for i=1:n
 [param,  Er(i)]=model_kth(A_2(:,1),A_2(:,2),i);
 end
 figure
 plot(Er,'Linewidth',2)
 %% Cazul apartamentelor cu 2 camere EFG approach
  A_2=[52 61; 39 52; 51 68; 52 52; 58 59; 50 60; 45 60;  56 62; 50 60; 55 63; 46 66; 54 59; 58 65; 55 63];
  
  %for r=5:2:25
  r=12;
  output=zeros(1,max(A_2(:,1))-min(A_2(:,1))+1);
  for i=1:max(A_2(:,1))-min(A_2(:,1))+1
      [output(i)]=model_liniar_EFG(A_2(:,1),A_2(:,2),i+min(A_2(:,1)-1),r);
  end
  size(output)
  xx=min(A_2):max(A_2);
  size(xx)
  figure
  plot(A_2(:,1),A_2(:,2),'o','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',3)
  hold on
  plot(xx, output,'b','Linewidth',3);
  title('Locally weighted linear regression ');
  legend('Data','LWR')
  %end
  %% %% Cazul apartamentelor cu 2 camere, Tikhonov regularization
  clear all
  close all
  A_2=[52 61; 39 52; 51 68; 52 52; 58 59; 50 60; 45 60;  56 62; 50 60; 55 63; 46 66; 54 59; 58 65; 55 63];
     
  %64 115; 41 50; 59 78; 53 100; 75 85; 52 61; 54 61; 52 65; 51 57; 50 62; 49 65; 40 55; 54 95; 45 52; 54 83;...
   
  %37 51; 67 119; 65 90; 40 52; 57 82; 55 60; 80 125];
  n=12;
  k=16;

  
 Er=zeros(1,k);
 for i=1:2:k
  lambda=10^(i-1)-1;
 [param,  Er(i)]=model_kth_reg(A_2(:,1),A_2(:,2),lambda,n);
 end
 %end
 figure
 plot(Er,'Linewidth',2)