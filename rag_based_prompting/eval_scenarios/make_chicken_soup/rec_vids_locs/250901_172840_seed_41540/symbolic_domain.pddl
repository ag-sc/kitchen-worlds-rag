(define (domain mobile-robot-tamp)
  (:requirements :strips :equality)

  (:constants
    @movable @bottle @edible @medicine
  )



  (:predicates

    (Sink ?r)
    (Stove ?r)
    (Counter ?r)
    (Table ?r)
    (Salter ?o)
    (Egg ?o)
    (Veggie ?o)
    (Plate ?o)

    (Arm ?a)
    (Drawer ?o) ;;
    (Door ?o) ;;
    (Knob ?o) ;;
    (Joint ?o)
    (JointAffectLink ?j ?l)
    (UnattachedJoint ?o)

    (Edible ?o)
    (CleaningSurface ?s)
    (HeatingSurface ?s)
    (ControlledBy ?s ?n)




    (Controllable ?o)
    (Graspable ?o)
    (MovableLink ?o)
    (StaticLink ?o)
    (Stackable ?o ?r)
    (Containable ?o ?r)








    (HandleGrasped ?a ?o)  ;; released the handle
    (HandEmpty ?a)

    (CanMoveBase)
    (CanMove)
    (CanPull ?a)
    (CanUngrasp)
    (CanPick)
    (CanGraspHandle)

    (Cleaned ?o)
    (Cooked ?o)
    (Seasoned ?o)
    (Served ?o ?o2)
    (EnableOmelette ?egg1 ?veggie1 ?plate1)
    (ExistOmelette ?env1)

    (OpenedJoint ?o) ;;
    (ClosedJoint ?o) ;;
    (GraspedHandle ?o) ;;

    (On ?o ?r)
    (In ?o ?r) ;;
    (Holding ?a ?o)
    (Space ?r)


    (Debug1)
    (Debug2)
    (Debug3)


    ;; making planning more efficient
    (Picked ?o)
    (Placed ?o)
    (Pulled ?o)
    (PulledOneAction ?o)

    (Enabled)
    (Disabled)

    (increase) ; pddlgym.parser


  ;;----------------------------------------------------------------------
  ;;      extended predicates from _cooking_domain.pddl
  ;;----------------------------------------------------------------------


    (Region ?o)
    (Sprinkler ?o)
    (Food ?o)
    (SprinkledTo ?o1 ?o2)


  ;;----------------------------------------------------------------------
  ;;      extended predicates from _arrange_domain.pddl
  ;;----------------------------------------------------------------------


    (Moved ?o)
    (Stacked ?o ?r)


  ;;----------------------------------------------------------------------
  ;;      extended predicates from _pull_domain.pddl
  ;;----------------------------------------------------------------------





  )

  (:functions
    (PickCost)
    (PlaceCost)
  )

(:action move_base
    :parameters ()
    :precondition (and (CanMoveBase) (CanMove) 
                       )
    :effect (and 
                  (not (CanMove))
                 (increase (total-cost) 1)
            )
  )

(:action pick_from_supporter
    :parameters (?a ?o1 ?o2)
    :precondition (and (Graspable ?o1) (MovableLink ?o2) (CanPick)
                         (HandEmpty ?a) 
                       (not (Picked ?o1))
                       )
    :effect (and (CanMove) (Picked ?o1)
                  (not (HandEmpty ?a))
                 (increase (total-cost) 1)
            )
  )

(:action pick
    :parameters (?a ?o)
    :precondition (and (Graspable ?o) (CanPick)
                        (HandEmpty ?a) 
                       )
    :effect (and (CanMove) (Picked ?o)
                  (not (HandEmpty ?a))
                 (increase (total-cost) 1)
            )
  )

(:action place_to_supporter
    :parameters (?a ?o1 ?o2)
    :precondition (and (Graspable ?o1) (MovableLink ?o2)
                       (not (CanMove))
                       )
    :effect (and (HandEmpty ?a) (CanMove)
                 (increase (total-cost) 1)
            )
  )

(:action clean
    :parameters (?o ?r)
    :precondition (and (Stackable ?o ?r) (Sink ?r) (On ?o ?r))
    :effect (and (Cleaned ?o))
  )

(:action cook
    :parameters (?o ?r)
    :precondition (and (Stackable ?o ?r) (Stove ?r) (On ?o ?r)
                       (Cleaned ?o)
                       )
    :effect (and (Cooked ?o)
                 (not (Cleaned ?o))
                 )
  )

(:action season
    :parameters (?o ?r ?o2)
    :precondition (and (Stackable ?o ?r) (Counter ?r)
                       (On ?o ?r) (Cooked ?o)
                       (Stackable ?o2 ?r) (Salter ?o2)
                       (On ?o2 ?r))
    :effect (and (Seasoned ?o))
  )

(:action serve
    :parameters (?o ?r ?o2)
    :precondition (and (Stackable ?o ?r) (Table ?r)
                       (On ?o ?r) (Seasoned ?o)
                       (Stackable ?o2 ?r) (Plate ?o2)
                       (On ?o2 ?r))
    :effect (and (Served ?o ?o2))
  )

(:action just-clean
    :parameters (?a ?o ?s)
    :precondition (and (Controllable ?a) (HandEmpty ?a) 
                       (Stackable ?o ?s) (On ?o ?s)
                       (CleaningSurface ?s)
                       )
    :effect (and (Cleaned ?o) ) 
  )

(:action just-cook
    :parameters (?a ?o ?s)
    :precondition (and (Controllable ?a) (HandEmpty ?a) 
                       (Stackable ?o ?s) (On ?o ?s)
                       (HeatingSurface ?s) (Cleaned ?o)
                       )
    :effect (and (Cooked ?o)) 
  )

(:action just-serve
    :parameters (?a ?o ?r)
    :precondition (and (Controllable ?a) (HandEmpty ?a) 
                       (Stackable ?o ?r) (On ?o ?r) (Plate ?r)
                       (Cleaned ?o)
                       )
    :effect (and (Served ?o ?r))
  )

(:action wait-clean
    :parameters (?o ?s ?n)
    :precondition (and (Edible ?o) (CleaningSurface ?s) (ControlledBy ?s ?n)
                       (On ?o ?s) (GraspedHandle ?n)
                       )
    :effect (and (Cleaned ?o))
  )

(:action wait-cook
    :parameters (?o ?s ?n)
    :precondition (and (Edible ?o) (HeatingSurface ?s) (ControlledBy ?s ?n)
                       (On ?o ?s) (GraspedHandle ?n)
                       (Cleaned ?o)
                       )
    :effect (and (Cooked ?o)
                 (not (Cleaned ?o)))
  )

(:action sprinkle
    :parameters (?a ?o1 ?o2)
    :precondition (and (Sprinkler ?o1) (Region ?o2)
                       (not (CanMove))
                   )
    :effect (and (SprinkledTo ?o1 ?o2) (CanMove)
                 (increase (total-cost) 1)
            )
  )

(:action arrange
    :parameters (?a ?o ?r)
    :precondition (and (Graspable ?o) 
                       (not (CanMove))
                       (not (Stacked ?o ?r))  
                       )
    :effect (and (HandEmpty ?a) (CanPull ?a) (CanMove)
                 (Stacked ?o ?r) (Moved ?o)  
                 (Placed ?o)
                 (increase (total-cost) 1)
            )
  )

(:action grasp_pull_ungrasp_handle
   :parameters (?a ?o)
   :precondition (and (Joint ?o) (CanGraspHandle)
                      (not (PulledOneAction ?o)) (not (Pulled ?o))
                       (CanPull ?a) (HandEmpty ?a)
                      (UnattachedJoint ?o)
                    )
    :effect (and 
                 (PulledOneAction ?o) (GraspedHandle ?o) (CanMove)
            )
  )

(:action grasp_pull_ungrasp_handle_with_link
   :parameters (?a ?o ?l)
   :precondition (and (Joint ?o) (CanGraspHandle)
                       (CanPull ?a) (HandEmpty ?a)
                      (JointAffectLink ?o ?l)   
                    )
   :effect (and 
                (PulledOneAction ?o) (GraspedHandle ?o) (CanMove)
           )
  )


)