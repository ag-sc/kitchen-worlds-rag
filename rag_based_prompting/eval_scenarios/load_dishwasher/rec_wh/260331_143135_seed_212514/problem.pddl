
(define
  (problem test_kitchen_dishwasher_260331_143135_seed_212514)
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

	(arm left)
	(arm right)

	(canmovebase)

	(canpull right)
	(canpull left)

	(cangrasphandle)
	(graspable fork)
	(graspable braiserlid#1)
	(graspable pepper-shaker)
	(graspable chicken-leg)
	(graspable braiserbody#1)
	(graspable salt-shaker)
	(handempty left)
	(handempty right)

	(food chicken-leg)

	(controllable right)
	(controllable left)

	(space braiserbody#1)
	(space counter#1::indigo_drawer_top)
	(space counter#1::sektion)

	(region braiserbody#1)
	(region counter#1::indigo_tmp)
	(region counter#1::front_left_stove)
	(region counter#1::sektion)
	(region basin#1::basin_bottom)
	(region counter#1::front_right_stove)
	(region counter#1::hitman_countertop)
	(region braiserbody#1::braiser_bottom)
	(region counter#1::indigo_drawer_top)
	(region fridge#1::shelf_top)

	(sprinkler pepper-shaker)
	(sprinkler salt-shaker)

	(surface counter#1::indigo_tmp)
	(surface counter#1::front_left_stove)
	(surface basin#1::basin_bottom)
	(surface braiserbody#1::braiser_bottom)
	(surface counter#1::front_right_stove)
	(surface counter#1::hitman_countertop)
	(surface braiserbody#1)
	(surface fridge#1::shelf_top)

	(staticlink basin#1::basin_bottom)
	(staticlink counter#1::front_right_stove)
	(staticlink counter#1::sektion)
	(staticlink counter#1::hitman_countertop)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink braiserbody#1)
	(staticlink counter#1::indigo_tmp)
	(staticlink fridge#1::shelf_top)
	(staticlink counter#1::front_left_stove)

	(bconf q944=(2.0, 6.25, 0.2, 3.142))

	(atbconf q944=(2.0, 6.25, 0.2, 3.142))

	(door counter#1::chewie_door_right_joint)
	(door fridge#1::fridge_door)
	(door counter#1::chewie_door_left_joint)

	(joint counter#1::chewie_door_right_joint)
	(joint oven#1::knob_joint_3)
	(joint counter#1::indigo_drawer_top_joint)
	(joint oven#1::knob_joint_2)
	(joint counter#1::chewie_door_left_joint)
	(joint faucet#1::joint_faucet_0)
	(joint fridge#1::fridge_door)
	(movablelink counter#1::indigo_drawer_top)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint oven#1::knob_joint_3)
	(unattachedjoint oven#1::knob_joint_2)
	(unattachedjoint counter#1::chewie_door_left_joint)

	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)
	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto counter#1::indigo_drawer_top_joint counter#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(position fridge#1::fridge_door pstn1167=0.0)
	(position counter#1::chewie_door_right_joint pstn1165=0.0)
	(position counter#1::chewie_door_left_joint pstn1166=0.0)
	(position oven#1::knob_joint_2 pstn1168=0.0)
	(position faucet#1::joint_faucet_0 pstn1170=0.0)
	(position counter#1::indigo_drawer_top_joint pstn1164=0.0)
	(position oven#1::knob_joint_3 pstn1169=0.0)
	(stackable chicken-leg basin#1::basin_bottom)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable fork counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable braiserlid#1 braiserbody#1)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 counter#1::indigo_tmp)

	(containable braiserbody#1 counter#1::sektion)
	(containable chicken-leg counter#1::indigo_drawer_top)
	(containable salt-shaker counter#1::sektion)
	(containable fork counter#1::sektion)
	(containable pepper-shaker counter#1::indigo_drawer_top)
	(containable chicken-leg counter#1::sektion)
	(containable braiserlid#1 counter#1::indigo_drawer_top)
	(containable braiserbody#1 counter#1::indigo_drawer_top)
	(containable salt-shaker counter#1::indigo_drawer_top)
	(containable pepper-shaker counter#1::sektion)
	(containable fork counter#1::indigo_drawer_top)
	(containable chicken-leg braiserbody#1)
	(containable braiserlid#1 counter#1::sektion)

	(atposition faucet#1::joint_faucet_0 pstn1170=0.0)
	(atposition fridge#1::fridge_door pstn1167=0.0)
	(atposition counter#1::indigo_drawer_top_joint pstn1164=0.0)
	(atposition oven#1::knob_joint_3 pstn1169=0.0)
	(atposition counter#1::chewie_door_right_joint pstn1165=0.0)
	(atposition counter#1::chewie_door_left_joint pstn1166=0.0)
	(atposition oven#1::knob_joint_2 pstn1168=0.0)

	(pose chicken-leg p573=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(pose counter#1::indigo_drawer_top lp44=(0.594, 8.843, 0.722, 0.0, -0.0, 0.0))
	(pose salt-shaker p574=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(pose braiserbody#1 p570=(0.7, 8.855, 0.923, 0.0, -0.0, 1.571))
	(pose pepper-shaker p575=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose braiserlid#1 p571=(0.7, 8.855, 0.953, 0.0, -0.0, 1.571))

	(atpose salt-shaker p574=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(atpose braiserbody#1 p570=(0.7, 8.855, 0.923, 0.0, -0.0, 1.571))
	(atpose braiserlid#1 p571=(0.7, 8.855, 0.953, 0.0, -0.0, 1.571))
	(atpose chicken-leg p573=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(atpose counter#1::indigo_drawer_top lp44=(0.594, 8.843, 0.722, 0.0, -0.0, 0.0))
	(atpose pepper-shaker p575=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))

	(isclosedposition counter#1::chewie_door_right_joint pstn1165=0.0)
	(isclosedposition oven#1::knob_joint_3 pstn1169=0.0)
	(isclosedposition counter#1::indigo_drawer_top_joint pstn1164=0.0)
	(isclosedposition counter#1::chewie_door_left_joint pstn1166=0.0)
	(isclosedposition fridge#1::fridge_door pstn1167=0.0)
	(isclosedposition oven#1::knob_joint_2 pstn1168=0.0)
	(isclosedposition faucet#1::joint_faucet_0 pstn1170=0.0)

	(aconf right aq952=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(aconf left aq184=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(ataconf right aq952=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(ataconf left aq184=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(jointaffectlink counter#1::indigo_drawer_top_joint counter#1::indigo_drawer_top)
	(supported braiserlid#1 p571=(0.7, 8.855, 0.953, 0.0, -0.0, 1.571) braiserbody#1)
	(supported braiserbody#1 p572=(0.7, 8.855, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)

	(startpose counter#1::indigo_drawer_top lp44=(0.594, 8.843, 0.722, 0.0, -0.0, 0.0))

	(contained pepper-shaker p575=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained salt-shaker p574=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained fork p576=(0.735, 8.831, 0.689, 0.0, -0.0, 2.447) counter#1::indigo_drawer_top)

	(relpose fork rp44=(0.141, -0.012, -0.033, 0.0, -0.0, 2.447) counter#1::indigo_drawer_top)

	(atrelpose fork rp44=(0.141, -0.012, -0.033, 0.0, -0.0, 2.447) counter#1::indigo_drawer_top)

  )

  (:goal (and
    (open counter#1::chewie_door_left_joint)
  ))
)
        