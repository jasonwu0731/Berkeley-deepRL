
# Run
*   run behavioral cloning
```
$ python run_BC.py --grid
```
*   run DAgger
```
$ python run_DAgger.py 
```

# Result
### vanilla behavioral cloning
###### Succesful example
On task Hopper-v1, the behavioral

  Hopper-v1      |      expert |   imitation
-----------------|-------------|------------                 
mean reward      |  3779.344349| 3776.911487
std reward       |     3.122555| 3.245903

### DAgger
Result, presented for task `Hopper-v1` for which the DAgger policy outperforms vanilla.
![reward vs epoch](/imgs/dagger-vanilla-comp-Hopper-v1.png)