
# coding: utf-8

# In[1]:


import nltk
from nltk.corpus import brown
import tools as mt
import json
import os


# In[2]:


def ex_constitution_corpus():
    themes = {"news":["news","reviews","editorial"],
              "literature":["science_fiction","romance","fiction","mystery"],
              "sciences":["learned"]}
    nb_instances = 0
    corpus ={}
    for category in themes:
        print("category : ",category)
        nb_doc = len(brown.fileids(categories = themes[category]))
        print("Il y a ",nb_doc," %s(category) documents in Brown corpus."%category)
        nb_instances+=nb_doc
        corpus[category]=brown.fileids(categories = themes[category])
    print("Il y a ",nb_instances," documents dans les 3 catégories (news,literature,sciences) pour Brown.")
    return corpus

def get_train_test_corpus(corpus):
    import random
    train={}
    test={}
    for category,fileids in corpus.items():
        x = int(20*len(fileids)/100)# trier par hasard 20%
        test[category]=[]
        print("On prend %s éléments sur %s pour le test"%(str(x),str(len(fileids))))
        for i in range(x):
            index_doc = random.randint(0,len(fileids)-1)
            test[category].append(fileids[index_doc])
            fileids.remove(fileids[index_doc])
        train[category]=fileids
    dataset={"train":train,"test":test}
    return dataset
corpus=ex_constitution_corpus()
test_train = get_train_test_corpus(corpus)
test_train_json=json.dumps(test_train,indent=2)
chemin="train_test.json"
mt.ecrire(test_train_json,chemin)


# In[3]:


import tools as mt
def get_features_light(liste_fichier):
    features_file={}
    for fileid in liste_fichier:
        features_file[fileid]={}
        stats_mots=mt.combien_de(brown.words(fileid))
        for feature,value in stats_mots.items():
            features_file[fileid][feature]=value
    print("->Features extraites:", list(features_file[fileid].keys())[:20],"...")
    return features_file

train_test = json.load(open("train_test.json"))
print("\nRécupération de la liste des fichiers")
liste_all_files = mt.get_all_files(train_test)
print("-> %s fichiers"%str(len(liste_all_files)))
print("\nExtraction des features")
features_by_file = get_features_light(liste_all_files) 
print("\nEcriture de la sortie JSON")
filename = "features_by_file.json"
mt.sauvegarder(features_by_file, filename)
print("-> %s"%filename)


# In[4]:


def get_entete_arff(feature_names, classes):
    """On crée l'entête du arff"""
    lignes_arff = ["@relation TRAIN_DATABASE\n"]#nom de la relation
    for name in feature_names:
        lignes_arff.append("@attribute %s numeric\n"%name)#noms des features
    lignes_arff.append("@attribute classes {%s}\n"%",".join(classes))#les classes
    return lignes_arff

def get_lignes_arff(feature_names, classes, features_by_file):
    lignes_arff = get_entete_arff(feature_names, classes)
    lignes_arff.append("@data\n\n")#on passe à la partie DATA
    for nom_classe, l_fichier in classes.items():
        for fileid in l_fichier:#Pour chaque fichier
            l_values = [features_by_file[fileid][name] for name in feature_names]
            l_values.append(nom_classe)#pour l'évaluation
            # La liste des features séparées par des virgules :
            ligne_values = ",".join([str(x) for x in l_values])
            lignes_arff.append(ligne_values)
    return lignes_arff
train_test =mt.lire_json("train_test.json")
all_fileids = mt.get_all_files(train_test)
features_by_file = mt.lire_json("features_by_file.json")
id_hasard = all_fileids[0]
feature_names = features_by_file[id_hasard].keys()
feature_sets = {"all_feature":feature_names}
def creationOfArff(feature_sets):
    try:
        os.makedirs("arff_files")#création du dossier pour les ranger
        print("Dossier 'arff_files' créé")
    except:
        print("Dossier 'arff_files' déjà créé")
        pass
    for feature_set_name, feature_list in feature_sets.items():
        print("\nFeature set : %s"%feature_set_name)
        for dataset, classes in train_test.items():# dataset:train test. classes: {news:[],science:[],literature:[]}
            print("  Processing %s set"%dataset)
            filename = "arff_files/%s__%s.arff"%(feature_set_name, dataset)
            lignes_arff = get_lignes_arff(feature_list, classes, features_by_file)# feature_list: dict_keys([nb_happy...])
            mt.ecrire("\n".join(lignes_arff), filename)
            print("    ->sortie = %s"%filename)
    return 
creationOfArff(feature_sets)

