#################################################################################################################################
# Nous définissons les fonctions utiles à l'aboutissement du projet

#################################################################################################################################
# Table des prédictions et vérités terrains dans le bon format

pred_truth = function(prediction, truth){
  x <- as.factor(prediction)
  y <- as.factor(truth)
  
  x.best.labels.and.scores <- data.frame(NULL)
  
  for(i in levels(x)){
    i.ind <- which(x == i)
    max.j.ind.in.i.ind <- 0
    for(j in levels(y)){
      # retrieve the number of elements from class x_i in y_j
      nb.ind.j.from.ind.i <- length(which(y[i.ind] == j))
      
      # select the best number whether it is the greatest of its j class
      if(nb.ind.j.from.ind.i > max.j.ind.in.i.ind){
        max.j.ind.in.i.ind <- nb.ind.j.from.ind.i
        x.best.labels.and.scores[i,1] <- j
        x.best.labels.and.scores[i,2] <- max.j.ind.in.i.ind
      }
    }
  }
  colnames(x.best.labels.and.scores) <- c("labels", "scores")
  
  for(i in levels(x)){
    # save the prior value to compare the prior diagonal sum after each move
    prior.x <- x
    prior.y <- y
    
    #print(x.best.labels.and.scores[x.best.labels.and.scores$labels==x.best.labels.and.scores[i,1], 2]
    # check if the score is the greatest in its column for every row containing its greatest score in this same column
    # check also whether it is already the perfect association
    if((x.best.labels.and.scores[i, 2] >= max(x.best.labels.and.scores[x.best.labels.and.scores$labels==x.best.labels.and.scores[i,1], 2])) & (x.best.labels.and.scores[i, 1] != i)){
      y.levels <- levels(y)
      
      y.levels[y.levels == i] <- "r"
      levels(y) <- y.levels #y[y == i] <- "r"
      
      y.levels[y.levels == x.best.labels.and.scores[i, 1]] <- i
      levels(y) <- y.levels #y[y == x.best.labels.and.scores[i, 1]] <- i
      
      y.levels[y.levels == "r"] <- x.best.labels.and.scores[i, 1]
      levels(y) <- y.levels #y[y == "r"] <- x.best.labels.and.scores[i, 1]
      
      # check whether Tr(x/y) == sum(diag) is descreasing and allow that only if the current label column of y contains one singular best value 
      # (which means one and only max value in the column)
      
      x.num <- as.numeric(x)
      y.num <- as.numeric(y)
      prior.x.num <- as.numeric(prior.x)
      prior.y.num <- as.numeric(prior.y)
      
      if(sum(x.num[x.num == y.num] == y.num[y.num == x.num]) < sum(prior.x.num[prior.x.num == prior.y.num] == prior.y.num[prior.y.num == prior.x.num]) & (length(x.best.labels.and.scores[x.best.labels.and.scores$labels==x.best.labels.and.scores[i,1], 1]) > 1)){
        x <- prior.x
        y <- prior.y
      }
      else{
        # update the best labels and scores table
        old.value <- x.best.labels.and.scores[i,1]
        x.best.labels.and.scores$labels[x.best.labels.and.scores$labels == i] <- "r"
        
        x.best.labels.and.scores$labels[x.best.labels.and.scores$labels == x.best.labels.and.scores$labels[i]] <- i
        x.best.labels.and.scores$labels[x.best.labels.and.scores$labels == "r"] <- old.value
      }
    }
  }
  return(list(prediction = as.factor(as.character(x)), truth = as.factor(as.character(y))))
}

#################################################################################################################################
# calcul une matrice de confusion en prenant compte des deux partition corriger 

compute_matrix_confusion = function(var){
  cfm = confusionMatrix(var$prediction,var$truth)
  return(cfm)
}

#################################################################################################################################
# Afficher correctement les n à m images, transposition possible (utile pour afficher correctement JAFFE)

show_images = function(data,from=1,to=10,transpose=FALSE){
  final=vector()
  dim_mat = dim(data)  
  for (variable in seq(from,to)) {
    b = matrix(data[variable,],nrow=sqrt(dim_mat[2]),ncol=sqrt(dim_mat[2]),byrow = T)
    final=cbind(final,b)  
  }
  if(transpose){rst <- as.raster(t(final),max=max(data))}
  else{rst <- as.raster(final,max=max(data))}
  raster::plot(rst)
}

#################################################################################################################################
# Routine pour afficher la Matrice de confusion avec 
# m : une matrice de confusion calculer avec
# name : nom du dataset
# methode : méthode d'apprentissage utilisé
# Retourne un graphique

ggplotConfusionMatrix <- function(m,name_dataset,methode_used){
  mytitle <- paste("Matrice de confusion pour",name_dataset,": Accuracy", percent_format()(m$overall[1]),
                   "Kappa", percent_format()(m$overall[2]),methode_used)
  data_c <-  mutate(group_by(as.data.frame(m$table), Reference ), percentage = 
                      percent(Freq/sum(Freq)))
  p <-
    ggplot(data = data_c,
           aes(x = Reference, y = Prediction)) +
    geom_tile(aes(fill = log(Freq)), colour = "white") +
    scale_fill_gradient(low = "white", high = "green") +
    geom_text(aes(x = Reference, y = Prediction, label = percentage)) +
    theme(legend.position = "none") +
    ggtitle(mytitle)
  return(p)
}

#################################################################################################################################
# Analyse en Composantes Pincipales
# original_dat : jeux de données sous forme de dataframe 
# mat_dat : jeux de données sous forme de matrice 
# name_dataset : nom du jeu de données
# Retourne la PCA calculé ainsi que le Graphique des indvidus avec les classes colorés et que le taux de variance par composantes

pca_analyse = function(original_dat,mat_dat,name_dataset){
  res = list()
  res.pca = PCA(mat_dat)
  res$pca = res.pca
  res$eig=fviz_eig(res.pca,choice = c("variance"),addlabels=TRUE,main = name_dataset)
  res$ind=fviz_pca_ind(res.pca,col.ind=as.factor(original_dat$y-1),labelsize=0,title=name_dataset)
  return(res)
}

#################################################################################################################################
# 5 méthodes de clustering calculer suivant un intervalle de classe et avec ou non une initialisation d'une ACP en prenant compte 2 pc
# Retourne l'objet NbClust pour chaque méthode

five_cluster_method = function(data_images,min_nc=6,max_nc=15,acp=FALSE){
  res = list()
  indices_vector = c("kl","ch","silhouette","gamma","gap","ratkowsky","gplus")
  if(acp == TRUE){
    pca = PCA(data_images,ncp = 2)
    data_images = pca$ind$coord            
  }
  res$kmeans = NbClust(data=data_images, min.nc = min_nc, max.nc = max_nc, method = "kmeans", index = indices_vector)
  res$cah_complete = NbClust(data=data_images, min.nc = min_nc, max.nc = max_nc, method = "complete", index = indices_vector) 
  res$cah_single = NbClust(data=data_images, min.nc = min_nc, max.nc = max_nc, method = "single", index = indices_vector) 
  res$cah_ward = NbClust(data=data_images, min.nc = min_nc, max.nc = max_nc, method = "ward.D2", index = indices_vector) 
  res$cah_average = NbClust(data=data_images, min.nc = min_nc, max.nc = max_nc, method = "average", index = indices_vector) 
  return(res)
} 

#################################################################################################################################
# Calcul de la matrice de confusion pour les 5 méthodes suivant un jeu de données mis en entrée

comparaison_truth = function(res.cluster,res.truth){
  cfm = list()
  cfm$kmeans = compute_matrix_confusion(pred_truth(res.cluster$kmeans$Best.partition, res.truth))
  cfm$cah_complete = compute_matrix_confusion(pred_truth(res.cluster$cah_complete$Best.partition, res.truth))
  cfm$cah_single = compute_matrix_confusion(pred_truth(res.cluster$cah_single$Best.partition, res.truth))
  cfm$cah_ward = compute_matrix_confusion(pred_truth(res.cluster$cah_ward$Best.partition, res.truth))
  cfm$cah_average = compute_matrix_confusion(pred_truth(res.cluster$cah_average$Best.partition, res.truth))
  return(cfm)
}

#################################################################################################################################
# On compare la matrice de confusion avec les résultats des clustering avec le jeu de données initiales et celui avec l'ACP

comparaison_cluster = function(res.normal,res.pca){
  cfm = list()
  cfm$kmeans = compute_matrix_confusion(pred_truth(res.pca$kmeans$Best.partition, res.normal$kmeans$Best.partition))
  cfm$cah_complete = compute_matrix_confusion(pred_truth(res.pca$cah_complete$Best.partition, res.normal$cah_complete$Best.partition))
  cfm$cah_single = compute_matrix_confusion(pred_truth(res.pca$cah_single$Best.partition, res.normal$cah_single$Best.partition))
  cfm$cah_ward = compute_matrix_confusion(pred_truth(res.pca$cah_ward$Best.partition, res.normal$cah_ward$Best.partition))
  cfm$cah_average = compute_matrix_confusion(pred_truth(res.pca$cah_average$Best.partition, res.normal$cah_average$Best.partition))
  return(cfm)
}

#################################################################################################################################
# Calcul des différents graphique de matrice de confusion suivant les 5 méthodes et le jeu de données correspondant

cfm_plot_dataset = function(cfm_all,name_dataset){
  plt_all = list()
  plt_all$kmeans = ggplotConfusionMatrix(cfm_all$kmeans,name_dataset,"kmeans")
  plt_all$cah_complete = ggplotConfusionMatrix(cfm_all$cah_complete,name_dataset,"cah_complete")
  plt_all$cah_single = ggplotConfusionMatrix(cfm_all$cah_single,name_dataset,"cah_single")
  plt_all$cah_ward = ggplotConfusionMatrix(cfm_all$cah_ward,name_dataset,"cah_ward")
  plt_all$cah_average = ggplotConfusionMatrix(cfm_all$cah_average,name_dataset,"cah_average")  
  return(plt_all)
}

#################################################################################################################################
# Génére un tableau incluant les résultats des différentes méthodes suivant une initialisation avec le jeu de données de départ ou l'ACP
# Inclus l'accuracy et Kappa

generate_arr_comparaison = function(res.normal,res.pca,dataset){
  resultats<-data.frame(accuracy_normal=c(res.normal$kmeans$overall[1]*100,res.normal$cah_complete$overall[1]*100,res.normal$cah_single$overall[1]*100,res.normal$cah_ward$overall[1]*100,res.normal$cah_average$overall[1]*100),
                        accuracy_pca=c(res.pca$kmeans$overall[1]*100,res.pca$cah_complete$overall[1]*100,res.pca$cah_single$overall[1]*100,res.pca$cah_ward$overall[1]*100,res.pca$cah_average$overall[1]*100),
                        kappa_normal=c(res.normal$kmeans$overall[2]*100,res.normal$cah_complete$overall[2]*100,res.normal$cah_single$overall[2]*100,res.normal$cah_ward$overall[2]*100,res.normal$cah_average$overall[2]*100),
                        kappa_pca=c(res.pca$kmeans$overall[2]*100,res.pca$cah_complete$overall[2]*100,res.pca$cah_single$overall[2]*100,res.pca$cah_ward$overall[2]*100,res.pca$cah_average$overall[2]*100),
                        row.names=c("kmeans","cah_complete","cah_single","cah_ward","cah_average"))
  resultats$name = dataset
  return(resultats)
}

#################################################################################################################################
# Génére un tableau incluant les résultats des différentes méthodes mclut et mixmod
# Inclus ma sensitivté et la spécificité ainsi que la nmi et ari
#

generate_arr_comparaison_mm = function(res.mclust,res.mixmod,dataset){
  resultats<-data.frame(ari=c(res.mclust$ari*100,res.mixmod$ari*100),
                        nmi=c(res.mclust$nmi*100,res.mixmod$nmi*100),
                        sensivity_tpr=c(res.mclust$sensivity_tpr,res.mixmod$sensivity_tpr),
                        specificity_tnr=c(res.mclust$specificity_tnr,res.mixmod$specificity_tnr),
                        row.names=c("mclust","mixmod"))
  resultats$name = dataset
  resultats$nbCluster = 10
  return(resultats)
}

#################################################################################################################################
# Comparaison des 5 méthodes lorsqu'on donne le vrai nombre de classes afin de voir les méthodes qui se distinctes

get_result_clustering = function(data,partition_data,name_dataset){
  q3_without_pca = five_cluster_method(data,10,10)
  q3_with_pca = five_cluster_method(data,10,10,acp=TRUE)
  test = comparaison_truth(q3_with_pca,partition_data)
  test_pca = comparaison_truth(q3_without_pca,partition_data)
  test_arr = generate_arr_comparaison(test,test_pca,name_dataset)
  return(test_arr)
}

#################################################################################################################################
# Calcul une ACP sur le nombre de composant voulu et retourne le résultat (matrice des individus)

data_from_pca = function(data_images,ncp=2){
  pca = PCA(data_images,ncp = ncp)
  data_images = pca$ind$coord
  return(data_images)
}

#################################################################################################################################
# Calcul du modèle de mélange avec mclust et Rmixmod
# Suivant un intervalle de classe donnés
# Strategy EM avec une initialisation SEMMax et un nombre à 200 afin de déterminer quel initilisation 
# apportera une convergance du maximum de vraisemblance grande et rapide
# Nombre d'esseis augmenter à 5 ainsi qu'un nombre d'itération à 500
# seed fixé afin de pouvoir réaliser une comparaison suivant les différents modèles

mixture_model_analyse = function(data_images,nb_min,nb_max){
  data_images = data_from_pca(data_images)
  res = list()
  res$mclust = Mclust(data=data_images,G=nb_min:nb_max)
  res$mixmodClust = mixmodCluster(data=as.data.frame(data_images),nbCluster=c(nb_min:nb_max)
                                  ,strategy= new("Strategy", algo="EM", initMethod="SEMMax",nbIterationInInit=200,
                                                 nbTry=5,nbIterationInAlgo=500),seed=42)
  return(res)
}    

#################################################################################################################################
# Calcul la NMI, l'ARI et le taux de bien classé et de mal classé pour un modèle de mélange avec mclust avec la vrai partition

get_result_mm_mclust = function(mclust_resultat,partition_model){
  res = list()
  res$pred_truth = pred_truth(as.matrix(mclust_resultat$mclust$classification),partition_model)
  res$cfm = compute_matrix_confusion(res$pred_truth)
  res$ari = ARI(res$pred_truth$prediction,res$pred_truth$truth)
  res$nmi = NMI(res$pred_truth$prediction,res$pred_truth$truth)
  res$sensivity_tpr = as.numeric(mean(res$cfm$byClass[,1]*100))
  res$specificity_tnr = as.numeric(mean(res$cfm$byClass[,2]*100))
  return(res)
}

#################################################################################################################################
# Calcul la NMI, l'ARI et le taux de bien classé et de mal classé pour un modèle de mélange avec mixmod avec la vrai partition

get_result_mm_mixmod = function(mixmod_resultat,partition_model){
  res = list()
  res$pred_truth = pred_truth(as.matrix(mixmod_resultat$mixmodClust@bestResult@partition),partition_model)
  res$cfm = compute_matrix_confusion(res$pred_truth)
  res$ari = ARI(res$pred_truth$prediction,res$pred_truth$truth)
  res$nmi = NMI(res$pred_truth$prediction,res$pred_truth$truth)
  res$sensivity_tpr = as.numeric(mean(res$cfm$byClass[,1]*100))
  res$specificity_tnr = as.numeric(mean(res$cfm$byClass[,2]*100))
  return(res)
}

#################################################################################################################################
# Analyse avec un ISOMAP en initialisation avec une ACP sur seulement 2 pc avec 10 voisins
# mat_dataset : jeu de données initiale
# name_dataset : nom du jeu données

isomap_analyse = function(mat_dataset,name_dataset){
  res = list()
  res$data_dim1to10_ISOMAP = Isomap(data=data_from_pca(mat_dataset$X,2), dims=1:10, k=10, plotResiduals=TRUE,verbose =F,mod=FALSE)
  res$data_isomap_d = Isomap(data=data_from_pca(mat_dataset$X,2),dims=2, k=10,mod=FALSE)
  plot(res$data_isomap_d$dim2,t='n', main=paste("ISOMAP",name_dataset),xlab="",ylab="")
  text(res$data_isomap_d$dim2)
  return(res)
}

#################################################################################################################################
# Visualisation avec t-SNE
# original_dat : dataframe du jeu de donnée
# mat_dat : matrice du jeu données prétraiter
# name_dataset : Nom du jeu données pour l'affichage en titre du plot
# Perplexity : Esimation proches du nombre de voisins que peut possèder un point 
# Theta : Angle 0.1 calcul plus rapide mais moins précis, et 0.9 lent, mais plus précis

tsne_analyse = function(original_dat,mat_dat,name_dataset,perplexity=50,theta=0.9){
  res = list()
  res.tsne = Rtsne(mat_dat, check_duplicates=FALSE, pca=FALSE, perplexity=perplexity, theta=theta, dims=2,pca_center=FALSE,num_threads=8)
  res$tsne = res.tsne
  v1.cols = as.numeric(as.factor(original_dat$y))-1
  plot(res.tsne$Y,t='n', main=paste("t-SNE",name_dataset),xlab="",ylab="")
  text(res.tsne$Y, labels = v1.cols,col=v1.cols)
  return(res)
}

#################################################################################################################################
# Visualisation avec t-SNE sur les obtenues grace aux modèles de mélanges Gaussian
# original_dat : dataframe du jeu de donnée
# mat_dat : matrice du jeu données prétraiter
# name_dataset : Nom du jeu données pour l'affichage en titre du plot
# Partition à partir d'un modèle de mélange executé
# Perplexity : Esimation proches du nombre de voisins que peut possèder un point 
# Theta : Angle 0.1 calcul plus rapide mais moins précis, et 0.9 lent, mais plus précis

tsne_analyse_mm = function(original_dat,mat_dat,name_dataset,partition,perplexity=50,theta=0.9){
  res = list()
  res.tsne = Rtsne(mat_dat, check_duplicates=FALSE, pca=FALSE, perplexity=perplexity, theta=theta, dims=2,pca_center=FALSE,num_threads=8)
  res$tsne = res.tsne
  v1.cols = as.numeric(as.factor(partition))-1
  plot(res.tsne$Y,t='n', main=paste("t-SNE",name_dataset),xlab="",ylab="")
  text(res.tsne$Y, labels = v1.cols,col=v1.cols)
  return(res)
}

#################################################################################################################################
# Visualisation avec ISOMAP
# mat_dataset : dataframe du jeu de données
# x : matrice du jeu de données réduit puisque dans notre cas on initiaise avec ACP
# Nom du jeu de données afin de l'afficher en titre du graphique

isomap_visualisation = function(mat_dataset,x,name_dataset){
  v1.cols = as.numeric(as.factor(mat_dataset$y))-1
  plot(x$dim2,t='n', main=paste("ISOMAP",name_dataset),xlab="",ylab="")
  text(x$dim2, labels = v1.cols,col=v1.cols)
}

#################################################################################################################################
# Entraine un autoencodeur avec un encodeur qui possède un espace latent de 2
# Globalement un MLP avec une Batch Normalization, fonctions d'activation ReLU
# Fonction coût MSE
# Optimizer Adam

train_autoencoder = function(dataset,batch_size,epochs,name_dataset){
  res = list()
  original_dim <- ncol(dataset)
  encoding_dim <- 2
  
  input_layer <- layer_input(shape = c(original_dim)) 
  
  encoder <- 
    input_layer %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_batch_normalization() %>% 
    layer_dropout(rate = 0.2) %>% 
    layer_dense(units = 32, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dense(units = 2) 
  
  decoder <- 
    encoder %>% 
    layer_dense(units = 8, activation = "relu") %>% 
    layer_batch_normalization() %>% 
    layer_dropout(rate = 0.2) %>% 
    layer_dense(units = 16, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 64, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dense(units = original_dim) 
  
  autoencoder_model <- keras_model(inputs = input_layer, outputs = decoder)
  
  autoencoder_model %>% compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics = c('accuracy')
  )
  
  history <-
    autoencoder_model %>%
    keras::fit(dataset,
               dataset,
               epochs=epochs,
               shuffle=TRUE
    )
  keras::save_model_weights_hdf5(object = autoencoder_model,filepath = paste('Data//',name_dataset,'.hdf5',sep=""),overwrite = TRUE)
  
  encoder_model <- keras_model(inputs = input_layer, outputs = encoder)
  
  encoder_model %>% keras::load_model_weights_hdf5(filepath = paste('Data//',name_dataset,'.hdf5',sep=""),skip_mismatch = TRUE,by_name = TRUE)
  
  encoder_model %>% compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics = c('accuracy')
  )
  
  res$autoencoder_model = autoencoder_model
  res$encoder_model = encoder_model
  res$history = history
  return(res)
}

#################################################################################################################################
# On récupère un encoder déja entrainer

get_encoder_pretrain = function(dataset,name_dataset){
  res = list()
  original_dim <- ncol(dataset)
  encoding_dim <- 2
  input_layer <- layer_input(shape = c(original_dim)) 
  encoder <- 
    input_layer %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_batch_normalization() %>% 
    layer_dropout(rate = 0.2) %>% 
    layer_dense(units = 32, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 16, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dropout(rate = 0.1) %>%
    layer_dense(units = 8, activation = "relu") %>%
    layer_batch_normalization() %>% 
    layer_dense(units = 2) 

  encoder_model <- keras_model(inputs = input_layer, outputs = encoder)
  
  encoder_model %>% keras::load_model_weights_hdf5(filepath = paste('Data//',name_dataset,'.hdf5',sep=""),skip_mismatch = TRUE,by_name = TRUE)
  
  encoder_model %>% compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics = c('accuracy')
  )
  res$encoder_model = encoder_model
  res$summary= summary(encoder_model)
  return(res)
}

#################################################################################################################################
# Retourne les données encodées sur 2 dimensions à partir d'un encoder keras compiler 
# et charger avec un modèle puis la matrice des données initiales

visualize_autoencoded = function(encoder_model,dataset){
  embeded_points <- 
    encoder_model %>% 
    keras::predict_on_batch(x = dataset)
  return(embeded_points)
}

#################################################################################################################################
# Installation et import des différents package utilise au projet
chooseCRANmirror(ind=29)

.libraryI = function(f){
  if(!require(f,character.only = T))install.packages(f,character.only=T,repos = "https://cran.univ-paris1.fr/")
  library(f,character.only = T)
}
libraryI = function(...){
  cc=list(...)
  if(is.list(cc[[1]])){
    cc=cc[1]
  }
  for(i in cc){
    .libraryI(i)
  }
}
libraryI(
  "R.matlab",
  "dplyr",
  "imager",
  "FactoMineR",
  "factoextra",
  "ggplot2",
  "fields",
  "grid",
  "reticulate",
  "devtools",
  "magick",
  "scales",
  "lle",
  "kohonen",
  "Rtsne",
  "NbClust",
  "caret",
  "e1071",
  "aricode",
  "Rmixmod",
  "mclust",
  "raster",
  "coop",
  "MASS",
  "EMMIXskew",
  "keras",
  "tensorflow",
  "EMMIXmfa"
)
if (!requireNamespace("BiocManager", quietly = TRUE))
  install.packages("BiocManager")
if(!require("RDRToolbox")) BiocManager::install("RDRToolbox", version = "3.8")
library("RDRToolbox")
library(rstudioapi)   
#install_keras()