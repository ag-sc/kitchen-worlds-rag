
(define
  (problem test_kitchen_dishwasher_260323_173031_seed_212514)
  (:domain domain)

  (:objects
	base
	base-torso
	basin#1
	basin#1::basin_bottom
	braiserbody#1
	braiserbody#1::braiser_bottom
	braiserlid#1
	chicken-leg
	counter#1
	counter#1::chewie_door_left_joint
	counter#1::chewie_door_right_joint
	counter#1::front_left_stove
	counter#1::front_right_stove
	counter#1::hitman_countertop
	counter#1::indigo_drawer_top
	counter#1::indigo_drawer_top_joint
	counter#1::indigo_tmp
	counter#1::sektion
	faucet#1
	faucet#1::joint_faucet_0
	fork
	fridge#1
	fridge#1::fridge_door
	fridge#1::shelf_top
	head
	left_arm
	left_gripper
	oven#1
	oven#1::knob_joint_2
	oven#1::knob_joint_3
	pepper-shaker
	right_arm
	right_gripper
	salt-shaker
	torso
  )

  (:init
	;; discrete facts (e.g. types, affordances)
	(canmove)
	(canpick)

	(arm right)
	(arm left)

	(canmovebase)

	(canpull right)
	(canpull left)

	(cangrasphandle)
	(handempty left)
	(handempty right)

	(food chicken-leg)

	(controllable right)
	(controllable left)

	(sprinkler salt-shaker)
	(sprinkler pepper-shaker)

	(graspable pepper-shaker)
	(graspable chicken-leg)
	(graspable braiserbody#1)
	(graspable salt-shaker)
	(graspable fork)
	(graspable braiserlid#1)

	(surface basin#1::basin_bottom)
	(surface braiserbody#1::braiser_bottom)
	(surface braiserbody#1)
	(surface counter#1::front_left_stove)
	(surface counter#1::indigo_tmp)
	(surface counter#1::front_right_stove)
	(surface counter#1::hitman_countertop)
	(surface fridge#1::shelf_top)

	(joint faucet#1::joint_faucet_0)
	(joint counter#1::chewie_door_right_joint)
	(joint fridge#1::fridge_door)
	(joint oven#1::knob_joint_2)
	(joint oven#1::knob_joint_3)
	(joint counter#1::indigo_drawer_top_joint)
	(joint counter#1::chewie_door_left_joint)

	(bconf q8=(2.0, 6.25, 0.2, 3.142))

	(atbconf q8=(2.0, 6.25, 0.2, 3.142))
	(space counter#1::indigo_drawer_top)
	(space braiserbody#1)
	(space counter#1::sektion)

	(region counter#1::front_right_stove)
	(region counter#1::hitman_countertop)
	(region counter#1::indigo_tmp)
	(region counter#1::sektion)
	(region fridge#1::shelf_top)
	(region basin#1::basin_bottom)
	(region counter#1::indigo_drawer_top)
	(region braiserbody#1::braiser_bottom)
	(region braiserbody#1)
	(region counter#1::front_left_stove)

	(door counter#1::chewie_door_left_joint)
	(door fridge#1::fridge_door)
	(door counter#1::chewie_door_right_joint)

	(staticlink counter#1::hitman_countertop)
	(staticlink counter#1::sektion)
	(staticlink fridge#1::shelf_top)
	(staticlink basin#1::basin_bottom)
	(staticlink counter#1::front_left_stove)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink braiserbody#1)
	(staticlink counter#1::indigo_tmp)
	(staticlink counter#1::front_right_stove)

	(movablelink counter#1::indigo_drawer_top)

	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable fork counter#1::indigo_tmp)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable braiserlid#1 braiserbody#1)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable chicken-leg basin#1::basin_bottom)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable braiserbody#1 counter#1::indigo_tmp)

	(unattachedjoint counter#1::chewie_door_left_joint)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint oven#1::knob_joint_3)
	(unattachedjoint oven#1::knob_joint_2)

	(containable salt-shaker counter#1::indigo_drawer_top)
	(containable braiserlid#1 counter#1::indigo_drawer_top)
	(containable pepper-shaker counter#1::sektion)
	(containable pepper-shaker counter#1::indigo_drawer_top)
	(containable chicken-leg braiserbody#1)
	(containable braiserbody#1 counter#1::sektion)
	(containable fork counter#1::sektion)
	(containable braiserbody#1 counter#1::indigo_drawer_top)
	(containable fork counter#1::indigo_drawer_top)
	(containable chicken-leg counter#1::sektion)
	(containable salt-shaker counter#1::sektion)
	(containable chicken-leg counter#1::indigo_drawer_top)
	(containable braiserlid#1 counter#1::sektion)

	(isclosedposition faucet#1::joint_faucet_0 pstn1108=0.0)
	(isclosedposition oven#1::knob_joint_2 pstn1106=0.0)
	(isclosedposition fridge#1::fridge_door pstn1105=0.0)
	(isclosedposition oven#1::knob_joint_3 pstn1107=0.0)
	(isclosedposition counter#1::chewie_door_right_joint pstn1103=0.0)
	(isclosedposition counter#1::indigo_drawer_top_joint pstn1102=0.0)
	(isclosedposition counter#1::chewie_door_left_joint pstn1104=0.0)
	(isjointto counter#1::indigo_drawer_top_joint counter#1)
	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)

	(position counter#1::chewie_door_left_joint pstn1104=0.0)
	(position oven#1::knob_joint_3 pstn1107=0.0)
	(position oven#1::knob_joint_2 pstn1106=0.0)
	(position fridge#1::fridge_door pstn1105=0.0)
	(position counter#1::chewie_door_right_joint pstn1103=0.0)
	(position faucet#1::joint_faucet_0 pstn1108=0.0)
	(position counter#1::indigo_drawer_top_joint pstn1102=0.0)

	(atposition counter#1::indigo_drawer_top_joint pstn1102=0.0)
	(atposition counter#1::chewie_door_left_joint pstn1104=0.0)
	(atposition oven#1::knob_joint_2 pstn1106=0.0)
	(atposition faucet#1::joint_faucet_0 pstn1108=0.0)
	(atposition oven#1::knob_joint_3 pstn1107=0.0)
	(atposition fridge#1::fridge_door pstn1105=0.0)
	(atposition counter#1::chewie_door_right_joint pstn1103=0.0)

	(pose braiserbody#1 p544=(0.7, 8.855, 0.923, 0.0, -0.0, 1.571))
	(pose counter#1::indigo_drawer_top lp33=(0.594, 8.843, 0.722, 0.0, -0.0, 0.0))
	(pose chicken-leg p547=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(pose braiserlid#1 p545=(0.7, 8.855, 0.953, 0.0, -0.0, 1.571))
	(pose pepper-shaker p549=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose salt-shaker p548=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))

	(atpose braiserbody#1 p544=(0.7, 8.855, 0.923, 0.0, -0.0, 1.571))
	(atpose braiserlid#1 p545=(0.7, 8.855, 0.953, 0.0, -0.0, 1.571))
	(atpose chicken-leg p547=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(atpose salt-shaker p548=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(atpose pepper-shaker p549=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(atpose counter#1::indigo_drawer_top lp33=(0.594, 8.843, 0.722, 0.0, -0.0, 0.0))

	(aconf right aq0=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(aconf left aq56=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(ataconf right aq0=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(ataconf left aq56=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(jointaffectlink counter#1::indigo_drawer_top_joint counter#1::indigo_drawer_top)

	(startpose counter#1::indigo_drawer_top lp33=(0.594, 8.843, 0.722, 0.0, -0.0, 0.0))

	(contained fork p550=(0.735, 8.831, 0.689, 0.0, -0.0, 2.447) counter#1::indigo_drawer_top)
	(contained salt-shaker p548=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained pepper-shaker p549=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)
	(relpose fork rp33=(0.141, -0.012, -0.033, 0.0, -0.0, 2.447) counter#1::indigo_drawer_top)
	(supported braiserbody#1 p546=(0.7, 8.855, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)
	(supported braiserlid#1 p545=(0.7, 8.855, 0.953, 0.0, -0.0, 1.571) braiserbody#1)

	(atrelpose fork rp33=(0.141, -0.012, -0.033, 0.0, -0.0, 2.447) counter#1::indigo_drawer_top)

  )

  (:goal (and
    (open counter#1::chewie_door_left_joint)
  ))
)
        