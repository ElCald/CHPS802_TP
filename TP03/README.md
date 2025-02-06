# TP03

### Execution `laplace3d.cu`

On remarque une accelération bien plus importante lors de l'utilisation du GPU qui prend ~25ms, face au CPU qui prend ~4000ms. <br>

#### Kernel
Le kernel calcul l'approximation de l'équation pour la case de la grille u1 du thread. <br>
Sous les conditions : 
    - Le thread se trouve dans la grille
    - Le thread ne se trouve pas au bord de la grille

#### Main
On initialise la grille qu'on copie dans la mémoire `device`. Puis on lance les blocs de threads un nomreb `REPEAT` de fois. À chaque fois on swap la grille input et output afin de recalculer la grille.

#### Vérification des grilles `laplace3d_test.cu`
Test unitaire sur les grilles `h_u1` et `h_u2`, cette dernière comporte les valeurs des calcules fait sur le GPU.

#### Tailles blocs et temps d'exécution `laplace3d.cu`
    - X=1, Y=1   : 20x GPU_laplace3d: 424.0 (ms) 
    - X=1, Y=32  : 20x GPU_laplace3d: 376.8 (ms)
    - X=32, Y=1  : 20x GPU_laplace3d: 49.9 (ms)
    - X=8, Y=8   : 20x GPU_laplace3d: 41.5 (ms)
    - X=8, Y=16  : 20x GPU_laplace3d: 34.5 (ms)
    - X=16, Y=16 : 20x GPU_laplace3d: 25.4 (ms)
    - X=16, Y=32 : 20x GPU_laplace3d: 26.0 (ms) 
    - X=24, Y=24 : 20x GPU_laplace3d: 25.4 (ms)
    - X=32, Y=32 : 20x GPU_laplace3d: 25.0 (ms) 

Plus on réduit le nombre de blocs et plus le temps d'exécution augmente. L'augmentation du nombre de bloc après 16, ne change pas le temps d'exécution. Nous sommes donc sur la version la plus optimisée en terme de taille de blocs avec une taille de 16*16.


#### Tailles blocs et temps d'exécution `laplace3d_new.cu`
    - X=1, Y=1, Z=1     : 20x GPU_laplace3d_new: 2049.1 (ms)
    - X=4, Y=1, Z=1     : 20x GPU_laplace3d_new: 543.5 (ms)
    - X=4, Y=4, Z=1     : 20x GPU_laplace3d_new: 169.8 (ms)
    - X=4, Y=4, Z=4     : 20x GPU_laplace3d_new: 59.1 (ms) 
    - X=8, Y=4, Z=4     : 20x GPU_laplace3d_new: 40.3 (ms)
    - X=8, Y=8, Z=4     : 20x GPU_laplace3d_new: 27.2 (ms)
    - X=8, Y=8, Z=8     : 20x GPU_laplace3d_new: 27.5 (ms)
    - X=16, Y=4, Z=4    : 20x GPU_laplace3d_new: 23.0 (ms)  
    - X=16, Y=8, Z=1    : 20x GPU_laplace3d_new: 35.4 (ms)
    - X=24, Y=1, Z=1    : 20x GPU_laplace3d_new: 121.0 (ms) 
    - X=4, Y=4, Z=16    : 20x GPU_laplace3d_new: 59.7 (ms)
    - X=4, Y=16, Z=4    : 20x GPU_laplace3d_new: 58.9 (ms)

Comme pour les tests effectués sur `laplace3d.cu`, une diminution du nombre de bloc entraine l'augmentation du temps d'exécution.