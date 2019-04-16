% training example for simple Go, where two neural networks are used
% (one for black and the other for white)
% written by Sae-Young Chung
% last update: 2016/4/13
% EE405C Network of Smart Systems, KAIST

% simple go
game1.winner=@winner_simple_go;
game1.valid=@valid_simple_go;
game1.init=@init_simple_go; % state needs to be initialized
game1.nx=5;  % board size nx*ny
game1.ny=5;
game1.name='simple go';

% 3x3 3-mok (tic tac toe)
game2.winner=@winner_n_mok;
game2.valid=@valid_basic;
game2.nx=3;     % board size nx * ny
game2.ny=3;
game2.n=3;      % specify n in n-mok
game2.name='tic tac toe';
game2.theme='basic';

% 9x9 5-mok
game3.winner=@winner_n_mok;
game3.valid=@valid_basic;
game3.nx=9;     % board size nx * ny
game3.ny=9;
game3.n=5;      % specify n in n-mok
game3.name='5 mok';

% othello
game4.winner=@winner_othello;
game4.valid=@valid_othello;
game4.init=@init_othello;  % in othello, initialization is needed since the initial board is not empty
game4.nx=8;
game4.ny=8;
game4.name='othello';
game4.theme='basic';
game=game1;      % specify which game to play
k=5;
load('rangen4.mat')

layers = [imageInputLayer([game.nx game.ny 3],'Normalization','none');
          convolution2dLayer(3,30,'Padding',1);
          reluLayer();
          convolution2dLayer(3,50,'Padding',1);
          reluLayer();
          convolution2dLayer(3,70,'Padding',1);
          reluLayer();
          fullyConnectedLayer(100);
          reluLayer();
          fullyConnectedLayer(3);
          softmaxLayer();
          classificationLayer()];

opts = trainingOptions('sgdm','Verbose',0,'InitialLearnRate',0.05,'MiniBatchSize',1024,'MaxEpochs',10);
% InitialLearnRate is changed from 0.05 to 0.15 by me
% MiniBatchSize is halved (1024->512) by me
tic
        n_train=80000;   % number of games to play for training
        n_test=1000;    % number of games to play for testing
        % play ng games between two players using the previous generation value network
        % introduce randomness in moves for robustness
        mt=floor(game.nx*game.ny/2);
        % 0.7 is added by me
        r1r=rand(n_train,1);
        r2r=rand(n_train,1);
        r1k=randi(mt*2,n_train,1);
        r2k=randi(mt*2,n_train,1);
        r1=(r1k>mt).*r1r+(r1k<=mt).*(-r1k)+0.1-(r1k>mt).*r1r+(r1k<=mt).*(-r1k)*0.1;
        r2=(r2k>mt).*r2r+(r2k<=mt).*(-r2k);
        [d1,w1,wp1,d2,w2,wp2]=play_games(game,net1{k-1},r1,net2{k-1},r2,n_train);

    % data augmentation
    [d1,w1]=data_augmentation(d1,w1);
    % train the next generation value network
    net1{k}=trainNetwork(d1,w1,layers,opts);
    save('tempa.mat','net1')    % save variables
    clear d1 w1   % to conserve memory
    % data augmentation
    [d2,w2]=data_augmentation(d2,w2);
    % train the next generation value network
    net2{k}=trainNetwork(d2,w2,layers,opts);
    save('tempb.mat','net2')    % save variables
    clear d2 w2   % to conserve memory
    toc

    disp(sprintf('Evaluating generation %d neural network', k))
    s=play_games(game,net1{k},0,[],1,n_test);
    win1(k)=s(1); loss1(k)=s(2); tie1(k)=s(3);
    disp(sprintf('  net plays black: win=%f, loss=%f, tie=%f', win1(k), loss1(k), tie1(k)))
    s=play_games(game,[],1,net2{k},0,n_test);
    win2(k)=s(2); loss2(k)=s(1); tie2(k)=s(3);
    disp(sprintf('  net plays white: win=%f, loss=%f, tie=%f', win2(k), loss2(k), tie2(k)))
    telapsed{k}=toc;
    toc
    disp(' ')
    save('rangen5','net1','net2','win1','loss1','tie1','win2','loss2','tie2','telapsed')
