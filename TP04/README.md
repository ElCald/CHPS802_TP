# TP04 - Réductions

### Vérfication d'un résultat correct avec test unitaire

Un test unitaire est fait avec le programme `reduction_test.cu`, on vérifie que la somme calculée sur le CPU est identique à celle calculée sur le GPU avec `ASSERT_FLOAT_EQ()`.


### Partage des threads adaptatif

Le partage se fait avec un nombre de threads qui est une puissance de 2, de ce fait le nombre d'élement devait aussi être cette puissance de 2 afin d'être parfaitement réparti. <br>

Dans le prgramme `reduction_adaptative.cu`, on trouve la puissance de 2 la plus proche de notre nombre d'éléments, c'est donc ce nombre de threads qui va travailler, le reste des éléments sera répartis sur les premiers threads avec un élément de plus pour chaque thread.


### Partage avec plusieurs blocs

Jusqu'à présent nous n'utilisions qu'un seul bloc, à présent dans le programme `reduction_multiblock.cu`, il est possible de choisir autant de bloc que nécéssaire.

Chaque thread dans son bloc calcul la somme qui lui est associé en plaçant le résultat dans la mémoire partagé du bloc. Enfin on effectue une réduction sur chacun des blocs dans le CPU.


### Partage en Shuffle

Programme appliquant la méthode shuffle : `reduction_shuffle.cu`.

Pour limiter les accès en mémoire partagée par les threads, on utilise la méthode shuffle qui permet à un thread de récupérer le résultat calculé par un autre thread après lui dans un même warp.