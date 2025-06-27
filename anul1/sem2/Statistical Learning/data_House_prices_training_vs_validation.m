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
 % figure
 % plot(A(:,1),A(:,2),'o','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',3)
 % title('Area vs House price')
 n=size(A,1);
 n_tr=n*0.8;
 n_v=n*0.2;
 permutare = randperm(n);

 A_tr=A(permutare(1:n_tr),:);
 A_v=A(permutare(n_tr+1:n),:);

  figure
 plot(A_tr(:,1),A_tr(:,2),'o','MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',3)
 hold on
 plot(A_v(:,1),A_v(:,2),'o','MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',3)
 legend('Training','Validation')
 title('Area vs House price Training vs Validation')

 k=12;% gradul maxim al polinomului de regresie
 Er_tr=zeros(1,k);
 Er_v=zeros(1,k);
 for i=1:k
 [param,  Er_tr(i)]=model_kth(A_tr(:,1),A_tr(:,2),i);
 hold on
 plot(A_v(:,1),A_v(:,2),'o','MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',3)

 xx=zeros(i+1,n_v);
    for j=1:i+1
        xx(j,:)=A_v(:,1).^(i+1-j);
    end
 Er_v(i)=(sum((param'*xx-A_v(:,2)').^2)/n_v)^(1/2);
 end
 figure
 plot(Er_tr,'-o','MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',3)
 hold on
 plot(Er_v,'-o','MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',3)
 legend('Training','Validation')
 title('RMSE models Training vs Validation')
 figure
 plot(Er_tr,'-o','MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',3)
  title('RMSE models Training ')

figure
 plot(Er_tr(1:6),'-o','MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',3)
 hold on
 plot(Er_v(1:6),'-o','MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',3)
 legend('Training','Validation')
 title('RMSE models Training vs Validation')




  %%
  %for r=5:2:25
  r=18;
  output_tr=zeros(1,max(A_tr(:,1))-min(A_tr(:,1))+1);
  for i=1:max(A_tr(:,1))-min(A_tr(:,1))+1
      [output(i)]=model_liniar_EFG(A_tr(:,1),A_tr(:,2),i+min(A_tr(:,1)-1),r);
  end
  size(output)
  xx=min(A_tr):max(A_tr);
  size(xx)
  figure
  plot(A_tr(:,1),A_tr(:,2),'o','MarkerEdgeColor','k','MarkerFaceColor','g','MarkerSize',3)
  hold on
  plot(A_v(:,1),A_v(:,2),'o','MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',3)
  plot(xx, output,'b','Linewidth',3);
  title('Locally weighted linear regression ');
  legend('Data','LWR')

  y_tr=zeros(n_tr,1);
  for i=1:n_tr
      [y_tr(i)]=model_liniar_EFG(A_tr(:,1),A_tr(:,2),A_tr(i,1),r);
  end
  Er2_tr=sqrt(1/n_tr*sum((A_tr(:,2)-y_tr).^2));
 y_v=zeros(n_v,1);
 for i=1:n_v
      [y_v(i)]=model_liniar_EFG(A_tr(:,1),A_tr(:,2),A_v(i,1),r);
  end
   Er2_v=sqrt(1/n_v*sum((A_v(:,2)-y_v).^2));
  %end
  %% %% Cazul apartamentelor cu 2 camere, Tikhonov/Ridge regularization


  n=6;%gradul polonomului
  k=16;%puterea maxima a lui lambda
 
 Er_tr_R=zeros(1,k);
 for i=1:k
  lambda=10^(i-1)-1;
 [param,  Er_tr_R(i)]=model_kth_reg(A_tr(:,1),A_tr(:,2),lambda,n);
  xx=zeros(n+1,n_v);
    for j=1:n+1
        xx(j,:)=A_v(:,1).^(i+1-j);
    end
 Er_v_R(i)=(sum((param'*xx-A_v(:,2)').^2)/n_v)^(1/2);
 end
 %end
 figure
 plot(Er_tr_R(1:7),'-o','MarkerEdgeColor','k','MarkerFaceColor','b','MarkerSize',3)
 hold on
 plot(Er_v_R(1:7),'-o','MarkerEdgeColor','k','MarkerFaceColor','r','MarkerSize',3)