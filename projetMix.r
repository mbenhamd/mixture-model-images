#################################################################################################################################
                                            #Projet de Mod�le de m�lange
#################################################################################################################################
# Affiche les titres de mani�re centr� dans un notebook
IRdisplay::display_html("
<style>
h1,h2,h3,h4,h5,h6 {
text-align:center
}
</style>
")

# Script contenant les diff�rents fonction utiles aux questions, permet d'appliquer les t�ches r�p�titives aux 5 diff�rents dataset
source(file="util.r")

#################################################################################################################################
# Import des 5 matlab fichiers

mat_usps=R.matlab::readMat("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/USPS.mat?raw=true")
mat_optdigits = R.matlab::readMat("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/Optdigits.mat?raw=true")
mat_jaffe = R.matlab::readMat("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/jaffe.mat?raw=true")
mat_MFEAT1 = R.matlab::readMat("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/MFEAT1.mat?raw=true")
mat_MNIST5 = R.matlab::readMat("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/MNIST5.mat?raw=true")

#################################################################################################################################
#  Conversion matrice des diff�rents jeux de donn�es ainsi qu'une mise � l'�chelle [0,1]


#################################################################################################################################
# Cr�ation d'une liste contenant les vraies partitions de chaque jeux de donn�es afin de faciliter les comparaison

truth_partition = list()
truth_partition$jaffe = mat_jaffe$y 
truth_partition$usps = mat_usps$y
truth_partition$optdigits = mat_optdigits$y
truth_partition$mfeat1 = mat_MFEAT1$y
truth_partition$mnist5 = mat_MNIST5$y

#################################################################################################################################
# Dimension de chaque dataset

print(dim(mat_optdigits_X))
print(dim(mat_usps_X))
print(dim(mat_jaffe_X))
print(dim(mat_MFEAT1_X))
print(dim(mat_MNIST5_X))

#################################################################################################################################
#  Nommbre d'obeservation pour chaque classe

mat_MNIST5$y %>% table()
mat_MFEAT1$y %>% table()
mat_jaffe$y %>% table()
mat_usps$y %>% table()
mat_optdigits$y %>% table()

#################################################################################################################################
# Affichage de 10 images contenu dans le jeu de donn�es

show_images(mat_MNIST5_X,1,10)
show_images(mat_MFEAT1_X,1,10)
show_images(mat_jaffe_X,1,10,transpose = TRUE)
show_images(mat_usps_X,1,10)
show_images(mat_optdigits_X,1,10)


#################################################################################################################################
# Calcul de sparsit� pour chaque jeu de donn�es

sparsity(mat_jaffe_X)*100
sparsity(mat_MFEAT1_X)*100
sparsity(mat_optdigits_X)*100
sparsity(mat_MNIST5_X)*100
sparsity(mat_usps_X)*100

#################################################################################################################################
# Calcul des 5 ACP

res.pca_analyse.optidigits = pca_analyse(mat_optdigits,mat_optdigits_X,"OPTIDIGITS")
res.pca_analyse.jaffe = pca_analyse(mat_jaffe,mat_jaffe_X,"JAFFE")
res.pca_analyse.mnist5 = pca_analyse(mat_MNIST5,mat_MNIST5_X,"MNIST5")
res.pca_analyse.usps = pca_analyse(mat_usps,mat_usps_X,"USPS")
res.pca_analyse.mfeat1 = pca_analyse(mat_MFEAT1,mat_MFEAT1_X,"MFEAT1")


#################################################################################################################################
# Visualisation des individus et de leurs classes sur les deux premiers plans factoriel

plot(res.pca_analyse.optidigits$ind)
plot(res.pca_analyse.mfeat1$ind)
plot(res.pca_analyse.mnist5$ind)
plot(res.pca_analyse.usps$ind)
plot(res.pca_analyse.jaffe$ind)

#################################################################################################################################
# Pourcentage d'information contenus dans chaque composantes principales

plot(res.pca_analyse.optidigits$eig)
plot(res.pca_analyse.mfeat1$eig)
plot(res.pca_analyse.mnist5$eig)
plot(res.pca_analyse.usps$eig)
plot(res.pca_analyse.jaffe$eig)

#################################################################################################################################
# Visualisation des diff�rents jeu de donn�es avec ISOMAP initilis� avec une ACP � 30 dimension et 10 voisins
# Il a fallu une dizaine d'heures pour pouvoir calculer ces 5 variables.

isomap_MNIST5_pca = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/isomap_MNIST5_pca.rds?raw=true")
isomap_optdigits_pca = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/isomap_MNIST5_pca.rds?raw=true")
isomap_usps_pca = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/isomap_usps_pca.rds?raw=true")
isomap_MFEAT1_pca = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/isomap_MFEAT1_pca.rds?raw=true")
isomap_jaffe_pca = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/isomap_jaffe_pca.rds?raw=true")

isomap_visualisation(mat_optdigits,x = isomap_optdigits_pca$data_isomap_d,name="OPTIDIGITS")
isomap_visualisation(mat_MNIST5,x = isomap_MNIST5_pca$data_isomap_d,name="MNIST5")
isomap_visualisation(mat_MFEAT1,x = isomap_MFEAT1_pca$data_isomap_d,name="MFEAT1")
isomap_visualisation(mat_usps,x = isomap_usps_pca$data_isomap_d,name="USPS")
isomap_visualisation(mat_jaffe,x = isomap_jaffe_pca$data_isomap_d,name="JAFFE")

#################################################################################################################################
# Visualisation avec t-SNE sur les donn�es d'origines

res.tsne_analyse.optidigits = tsne_analyse(mat_optdigits,mat_optdigits_X,"OPTIDIGITS",perplexity =10,theta=0.5)
res.tsne_analyse.mnist5 = tsne_analyse(mat_MNIST5,mat_MNIST5_X,"MNIST5",perplexity =50,theta=0.9)
res.tsne_analyse.mfeat1 = tsne_analyse(mat_MFEAT1,mat_MFEAT1_X,"MFEAT1",perplexity =10,theta=0.5)
res.tsne_analyse.usps = tsne_analyse(mat_usps,mat_usps_X,"USPS",perplexity =10,theta=0.5)
res.tsne_analyse.jaffe = tsne_analyse(mat_jaffe,mat_jaffe_X,"JAFFE",perplexity =10,theta=0.9)

#################################################################################################################################
# Clustering avec les 5 m�thodes (kmeans,cah ward, cah average, cah complete, cah single).
# Jeu de donn�es initiales, crit�re de silouette.
# Chargement des objet pr�-calculer pour des nombres de cluster allant de 8 � 12 acp � false 
# five_cluster_method(X,min_nc=8,max_nc=12,acp=FALSE)

fcm_normal_optdigits = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/fcm_normal_optdigits.rds?raw=true")
fcm_normal_jaffe = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/fcm_normal_jaffe.rds?raw=true")
fcm_normal_mfeat1 = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/fcm_normal_mfeat1.rds?raw=true")
fcm_normal_mnist5 = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/fcm_normal_mnist5.rds?raw=true")
fcm_normal_usps = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/fcm_normal_usps.rds?raw=true")

#################################################################################################################################
# Clustering avec les 5 m�thodes (kmeans,cah ward, cah average, cah complete, cah single).
# Jeu de donn�es avec subi une ACP et initialisation avec les deux premi�res composantes principales, crit�re de silouette.
# five_cluster_method(X,min_nc=8,max_nc=12,acp=TRUE)
# Chargement des objets pour des k allant de 8 � 12

fcm_normal_optdigits_pca = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/fcm_normal_optdigits_pca.rds?raw=true")
fcm_normal_jaffe_pca = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/fcm_normal_jaffe_pca.rds?raw=true")
fcm_normal_mfeat1_pca = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/fcm_normal_mfeat1_pca.rds?raw=true")
fcm_normal_mnist5_pca = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/fcm_normal_mnist5_pca.rds?raw=true")
fcm_normal_usps_pca = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/fcm_normal_usps_pca.rds?raw=true")

fcm_normal_usps_pca$kmeans$Best.nc
fcm_normal_usps_pca$cah_complete$Best.nc
fcm_normal_usps_pca$cah_single$Best.nc
fcm_normal_usps_pca$cah_ward$Best.nc
fcm_normal_usps_pca$cah_average$Best.nc

#################################################################################################################################


#################################################################################################################################
# Comparaisons des partitions avec 5 m�thodes suivants une initialisation ou non avec une ACP

gac_optdigits = get_result_clustering(mat_optdigits_X,truth_partition$optdigits,"OPTDIGITS")
gac_jaffe = get_result_clustering(mat_jaffe_X,truth_partition$jaffe,"JAFFE")
gac_mfeat1 = get_result_clustering(mat_MFEAT1_X,truth_partition$mfeat1,"MFEAT1")
gac_mnist5 = get_result_clustering(mat_MNIST5_X,truth_partition$mnist5,"MNIST5")
gac_usps = get_result_clustering(res.normal,truth_partition$usps,"USPS")

#################################################################################################################################
# Mod�le de m�langes avec les packages mclust et rmixmod pour chaque jeu de donn�es
# Avec le jeu de donn�es initiales, seul le package mclust a �tait utilis� � cause d'un memory leak de la part de rmixmod 
# lorsque l'on donne un intervalle de classe avec un jeu de donn�e.

mm_jaffe = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/mm_jaffe.rds?raw=true")
mm_mnist5 = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/mm_MNIST5.rds?raw=true")
mm_optdigits = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/mm_optdigits.rds?raw=true")
mm_usps = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/mm_usps.rds?raw=true")
mm_mfeat1 = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/mm_MFEAT1.rds?raw=true")

plot(mm_jaffe$mclust$BIC)
plot(mm_mnist5$mclust$BIC)
plot(mm_optdigits$mclust$BIC)
plot(mm_usps$mclust$BIC)
plot(mm_mfeat1$mclust$BIC)

#################################################################################################################################
# Mod�le de m�langes avec les packages mclust et rmixmod pour chaque jeu de donn�es
# Avec le jeu de donn�es ayant subi une ACP, on donne en entr�e les deux composantes principales.

mm_jaffe_pca_10 =  readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/mm_jaffe_pca.rds?raw=true")
mm_MFEAT1_pca_10 = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/mm_MFEAT1_pca.rds?raw=true")
mm_MNIST5_pca_10 = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/mm_MNIST5_pca.rds?raw=true")
mm_optdigits_pca_10 = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/mm_optdigits_pca.rds?raw=true")
mm_usps_pca_10 = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/mm_usps_pca.rds?raw=true")

# Plot pour la s�l�ction du mod�le avec le crit�re BIC (package mclust)
plot(mm_jaffe_pca_10$mclust$BIC)
plot(mm_MFEAT1_pca_10$mclust$BIC)
plot(mm_MNIST5_pca_10$mclust$BIC)
plot(mm_optdigits_pca_10$mclust$BIC)
plot(mm_usps_pca_10$mclust$BIC)

# Print pour la s�l�ction du mod�le avec le package mclust
print(paste("JAFFE",mm_jaffe_pca_10$mclust$modelName,mm_jaffe_pca_10$mclust$G))
print(paste("MFEAT",mm_MFEAT1_pca_10$mclust$modelName,mm_MFEAT1_pca_10$mclust$G))
print(paste("MNIST",mm_MNIST5_pca_10$mclust$modelName,mm_MNIST5_pca_10$mclust$G))
print(paste("OPTDIGITS",mm_optdigits_pca_10$mclust$modelName,mm_optdigits_pca_10$mclust$G))
print(paste("USPS",mm_usps_pca_10$mclust$modelName,mm_usps_pca_10$mclust$G))

# Pint pour la s�l�ction du mod�le avec le package mixmod
print(paste("JAFFE",mm_jaffe_pca_10$mixmodClust@bestResult@nbCluster,mm_jaffe_pca_10$mixmodClust@bestResult@model))
print(paste("MFEAT",mm_MFEAT1_pca_10$mixmodClust@bestResult@nbCluster,mm_MFEAT1_pca_10$mixmodClust@bestResult@model))
print(paste("MNIST",mm_MNIST5_pca_10$mixmodClust@bestResult@nbCluster,mm_MNIST5_pca_10$mixmodClust@bestResult@model))
print(paste("OPTDIGITS",mm_optdigits_pca_10$mixmodClust@bestResult@nbCluster,mm_optdigits_pca_10$mixmodClust@bestResult@model))
print(paste("USPS",mm_usps_pca_10$mixmodClust@bestResult@nbCluster,mm_usps_pca_10$mixmodClust@bestResult@model))

#################################################################################################################################
# MclustDR du package mclust
# blabla

plot(MclustDR(mm_mfeat1$mclust))
plot(MclustDR(mm_jaffe$mclust))
plot(MclustDR(mm_mnist5$mclust))
plot(MclustDR(mm_optdigits$mclust))
plot(MclustDR(mm_usps$mclust))

#################################################################################################################################
# Comparaison des r�sultats des mod�les de m�langes provenant de mclust et rmixmod avec k = 10

# Calcul avec le vrai nombre de classe sur les deux partitions
mma_jaffe = mixture_model_analyse(data_images = mat_jaffe_X,10,10)
mma_mfeat = mixture_model_analyse(data_images = mat_MFEAT1_X,10,10)
mma_mnist = mixture_model_analyse(data_images = mat_MNIST5_X,10,10)
mma_optdigits = mixture_model_analyse(data_images = mat_optdigits_X,10,10)
mma_usps = mixture_model_analyse(data_images = mat_usps_X,10,10)

# R�sultat avec mclust

res_final_jaffe_mclust = get_result_mm_mclust(mma_jaffe,truth_partition$jaffe)
res_final_mfeat1_mclust = get_result_mm_mclust(mma_mfeat,truth_partition$mfeat1)
res_final_optdigits_mclust = get_result_mm_mclust(mma_optdigits,truth_partition$optdigits)
res_final_mnist5_mclust = get_result_mm_mclust(mma_mnist,truth_partition$mnist5)
res_final_usps_mclust = get_result_mm_mclust(mma_usps,truth_partition$usps)

# R�sultat avec mixmod

res_final_jaffe_mixmod = get_result_mm_mixmod(mma_jaffe,truth_partition$jaffe)
res_final_mfeat1_mixmod = get_result_mm_mixmod(mma_mfeat,truth_partition$mfeat1)
res_final_optdigits_mixmod = get_result_mm_mixmod(mma_optdigits,truth_partition$optdigits)
res_final_mnist5_mixmod = get_result_mm_mixmod(mma_mnist,truth_partition$mnist5)
res_final_usps_mixmod = get_result_mm_mixmod(mma_usps,truth_partition$usps)

#################################################################################################################################
# On visualise les classes provenant des mod�les de m�lange gr�ce � la t-SNE

tsne_jaffe_mm = tsne_analyse_mm(original_dat = mat_jaffe,mat_dat = mma_jaffe$mixmodClust@data,mma_jaffe$mixmodClust@bestResult@partition,name_dataset = "jaffe mod�le de m�lange",perplexity = 10,theta = 0.9)
tsne_mnist5_mm = tsne_analyse_mm(original_dat = mat_MNIST5,mma_mnist$mixmodClust@data,mma_mnist$mixmodClust@bestResult@partition,name_dataset = "mnist5 mod�le de m�lange",perplexity = 50,theta = 0.9)
tsne_optdigits_mm = tsne_analyse_mm(original_dat = mat_optdigits,mma_optdigits$mixmodClust@data,mma_optdigits$mixmodClust@bestResult@partition,name_dataset = "optdigits mod�le de m�lange",perplexity = 50,theta = 0.9)
tsne_usps_mm = tsne_analyse_mm(original_dat = mat_usps,mma_usps$mixmodClust@data,mma_usps$mixmodClust@bestResult@partition,name_dataset = "usps mod�le de m�lange",perplexity = 50,theta = 0.9)
tsne_mfea_mm = tsne_analyse_mm(original_dat = mat_MFEAT1,mma_mfeat$mixmodClust@data,mma_mfeat$mixmodClust@bestResult@partition,name_dataset = "mfea mod�le de m�lange",perplexity = 60,theta = 0.9)

#################################################################################################################################
# On entraine sur les diff�rents jeux de donn�es les autoencodeurs   
# Inutile de le relancer si des mod�les sont d�ja pr�-entrainer et mis dans le dossier afin de gagner du temps lors de la relecture du script
# Un r�sum� de chaque param�tres
# train_jaffe = train_autoencoder(dataset = mat_jaffe_X,batch_size =32,epochs = 500,name_dataset = "jaffe")
# train_mnist5 = train_autoencoder(dataset = mat_MNIST5_X,batch_size =32,epochs = 500,name_dataset = "mnist5")
# train_optdigits = train_autoencoder(dataset = mat_optdigits_X,batch_size =32,epochs = 500,name_dataset = "optdigits")
# train_usps = train_autoencoder(dataset = mat_usps_X,batch_size =32,epochs = 500,name_dataset = "usps")
# train_mfea = train_autoencoder(dataset = mat_MFEAT1_X,batch_size =32,epochs = 500,name_dataset = "mfea")

train_jaffe = get_encoder_pretrain(dataset = mat_jaffe_X,name_dataset = "jaffe")
train_mnist5 = get_encoder_pretrain(dataset = mat_MNIST5_X,name_dataset = "mnist5")
train_optdigits = get_encoder_pretrain(dataset = mat_optdigits_X,name_dataset = "optdigits")
train_usps = get_encoder_pretrain(dataset = mat_usps_X,name_dataset = "usps")
train_mfea = get_encoder_pretrain(dataset = mat_MFEAT1_X,name_dataset = "mfea")

#################################################################################################################################
# On visualise les donn�es encod�es gr�ce � la t-SNE

tsne_jaffe = tsne_analyse(original_dat = mat_jaffe,mat_dat = visualize_autoencoded(train_jaffe$encoder_model,mat_jaffe_X),name_dataset = "jaffe",perplexity = 60,theta = 0.9)
tsne_mnist5 = tsne_analyse(original_dat = mat_MNIST5,mat_dat = visualize_autoencoded(train_mnist5$encoder_model,mat_MNIST5_X),name_dataset = "mnist5")
tsne_optdigits = tsne_analyse(original_dat = mat_optdigits,mat_dat = visualize_autoencoded(train_optdigits$encoder_model,mat_optdigits_X),name_dataset = "optdigits")
tsne_usps = tsne_analyse(original_dat = mat_usps,mat_dat = visualize_autoencoded(train_usps$encoder_model,mat_usps_X),name_dataset = "usps")
tsne_mfea = tsne_analyse(original_dat = mat_MFEAT1,mat_dat = visualize_autoencoded(train_mfea$encoder_model,mat_MFEAT1_X),name_dataset = "mfea")

#################################################################################################################################
# On entraine les mod�les de m�langes sur les donn�es encod�es

mm_jaffe_10_autoencoder = mixture_model_analyse(visualize_autoencoded(train_jaffe$encoder_model,mat_jaffe_X) ,10,10)
mm_MFEAT1_10_autoencoder = mixture_model_analyse(visualize_autoencoded(train_mfea$encoder_model,mat_MFEAT1_X) ,10,10)
mm_MNIST5_10_autoencoder = mixture_model_analyse(visualize_autoencoded(train_mnist5$encoder_model,mat_MNIST5_X) ,10,10)
mm_optdigits_10_autoencoder = mixture_model_analyse(visualize_autoencoded(train_optdigits$encoder_model,mat_optdigits_X) ,10,10)
mm_usps_10_autoencoder = mixture_model_analyse(visualize_autoencoded(train_usps$encoder_model,mat_usps_X) ,10,10)

# On r�cupere les r�sultats pour le mod�le de m�lange calculer gr�ce � Rmixmod

res_final_jaffe_autoencoder_mixmod = get_result_mm_mixmod(mm_jaffe_10_autoencoder,truth_partition$jaffe)
res_final_mfeat1_autoencoder_mixmod = get_result_mm_mixmod(mm_MFEAT1_10_autoencoder,truth_partition$mfeat1)
res_final_optdigits_autoencoder_mixmod = get_result_mm_mixmod(mm_optdigits_10_autoencoder,truth_partition$optdigits)
res_final_mnist5_autoencoder_mixmod = get_result_mm_mixmod(mm_MNIST5_10_autoencoder,truth_partition$mnist5)
res_final_usps_autoencoder_mixmod = get_result_mm_mixmod(mm_usps_10_autoencoder,truth_partition$usps)

# On r�cupere les r�sultats pour le mod�le de m�lange calculer gr�ce � mclust

res_final_jaffe_autoencoder_mclust = get_result_mm_mclust(mm_jaffe_10_autoencoder,truth_partition$jaffe)
res_final_mfeat1_autoencoder_mclust = get_result_mm_mclust(mm_MFEAT1_10_autoencoder,truth_partition$mfeat1)
res_final_optdigits_autoencoder_mclust = get_result_mm_mclust(mm_optdigits_10_autoencoder,truth_partition$optdigits)
res_final_mnist5_autoencoder_mclust = get_result_mm_mclust(mm_MNIST5_10_autoencoder,truth_partition$mnist5)
res_final_usps_autoencoder_mclust = get_result_mm_mclust(mm_usps_10_autoencoder,truth_partition$usps)

#################################################################################################################################
# EMMIX 

mfa_jaffe = readRDS("https://github.com/mbenhamd/mixture-model-images/blob/master/Data/mfa_jaffe.rds?raw=true")
