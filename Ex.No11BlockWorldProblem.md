# Ex.No: 11  Planning –  Block World Problem 
### DATE:                                                                            
### REGISTER NUMBER :212223060207
### DATE: 27/09/2025                                                                            
### AIM: 
To find the sequence of plan for Block word problem using PDDL  
###  Algorithm:
@@ -16,7 +16,35 @@ Step 9 : Define a problem for block world problem.<br>
Step 10 : Obtain the plan for given problem.<br> 

### Program:

```
(define (domain blocksworld)
(:requirements :strips :equality)
(:predicates (clear ?x)
             (on-table ?x)
             (arm-empty)
             (holding ?x)
             (on ?x ?y))
(:action pickup
  :parameters (?ob)
  :precondition (and (clear ?ob) (on-table ?ob) (arm-empty))
  :effect (and (holding ?ob) (not (clear ?ob)) (not (on-table ?ob)) 
               (not (arm-empty))))
(:action putdown
  :parameters  (?ob)
  :precondition (and (holding ?ob))
  :effect (and (clear ?ob) (arm-empty) (on-table ?ob) 
               (not (holding ?ob))))
(:action stack
  :parameters  (?ob ?underob)
  :precondition (and  (clear ?underob) (holding ?ob))
  :effect (and (arm-empty) (clear ?ob) (on ?ob ?underob)
               (not (clear ?underob)) (not (holding ?ob))))
(:action unstack
  :parameters(?ob?underob)
  :precondition (and (on ?ob ?underob) (clear ?ob) (arm-empty))
  :effect (and (holding ?ob) (clear ?underob)
               (not (on ?ob ?underob)) (not (clear ?ob)) (not (arm-empty)))))
```



@@ -26,8 +54,16 @@ Step 10 : Obtain the plan for given problem.<br>


### Input 
```
(define (problem pb1)
   (:domain blocksworld)
   (:objects a b)
   (:init (on-table a) (on-table b)  (clear a)  (clear b) (arm-empty))
   (:goal (and (on a b))))
```

### Output/Plan:
<img width="556" height="682" alt="image" src="https://github.com/user-attachments/assets/06840014-aa5d-416c-b81c-37f86f343472" />


