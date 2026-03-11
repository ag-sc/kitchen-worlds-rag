
(define
  (problem test_kitchen_chicken_soup_250904_083242_seed_989544)
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
	counter#1::indigo_tmp
	counter#1::sektion
	faucet#1
	faucet#1::joint_faucet_0
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
	(handempty left)
	(handempty right)

	(food chicken-leg)

	(controllable left)
	(controllable right)

	(region braiserbody#1)
	(region braiserbody#1::braiser_bottom)
	(region basin#1::basin_bottom)
	(region counter#1::front_right_stove)
	(region counter#1::sektion)
	(region counter#1::hitman_countertop)
	(region fridge#1::shelf_top)
	(region counter#1::indigo_tmp)
	(region counter#1::front_left_stove)

	(graspable chicken-leg)
	(graspable salt-shaker)
	(graspable braiserbody#1)
	(graspable braiserlid#1)
	(graspable pepper-shaker)

	(sprinkler pepper-shaker)
	(sprinkler salt-shaker)

	(space counter#1::sektion)
	(space braiserbody#1)

	(door fridge#1::fridge_door)
	(door counter#1::chewie_door_left_joint)
	(door counter#1::chewie_door_right_joint)

	(joint faucet#1::joint_faucet_0)
	(joint counter#1::chewie_door_right_joint)
	(joint fridge#1::fridge_door)
	(joint oven#1::knob_joint_3)
	(joint oven#1::knob_joint_2)
	(joint counter#1::chewie_door_left_joint)
	(staticlink fridge#1::shelf_top)
	(staticlink counter#1::indigo_tmp)
	(staticlink counter#1::front_left_stove)
	(staticlink braiserbody#1)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink basin#1::basin_bottom)
	(staticlink counter#1::front_right_stove)
	(staticlink counter#1::sektion)
	(staticlink counter#1::hitman_countertop)

	(bconf q832=(2.0, 6.25, 0.2, 3.142))

	(surface counter#1::front_left_stove)
	(surface braiserbody#1)
	(surface braiserbody#1::braiser_bottom)
	(surface basin#1::basin_bottom)
	(surface counter#1::front_right_stove)
	(surface counter#1::hitman_countertop)
	(surface fridge#1::shelf_top)
	(surface counter#1::indigo_tmp)

	(atbconf q832=(2.0, 6.25, 0.2, 3.142))

	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint oven#1::knob_joint_3)
	(unattachedjoint oven#1::knob_joint_2)
	(unattachedjoint counter#1::chewie_door_left_joint)

	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)
	(position oven#1::knob_joint_2 pstn18847=0.0)
	(position counter#1::chewie_door_left_joint pstn18845=-1.872)
	(position fridge#1::fridge_door pstn18846=1.78)
	(position faucet#1::joint_faucet_0 pstn18849=0.0)
	(position oven#1::knob_joint_3 pstn18848=0.0)
	(position counter#1::chewie_door_right_joint pstn18844=1.872)

	(containable pepper-shaker counter#1::sektion)
	(containable braiserbody#1 counter#1::sektion)
	(containable chicken-leg counter#1::sektion)
	(containable salt-shaker counter#1::sektion)
	(containable chicken-leg braiserbody#1)
	(containable braiserlid#1 counter#1::sektion)

	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable braiserlid#1 braiserbody#1)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable chicken-leg basin#1::basin_bottom)

	(isclosedposition oven#1::knob_joint_2 pstn18847=0.0)
	(isclosedposition faucet#1::joint_faucet_0 pstn18849=0.0)
	(isclosedposition oven#1::knob_joint_3 pstn18848=0.0)

	(atposition counter#1::chewie_door_left_joint pstn18845=-1.872)
	(atposition oven#1::knob_joint_3 pstn18848=0.0)
	(atposition fridge#1::fridge_door pstn18846=1.78)
	(atposition faucet#1::joint_faucet_0 pstn18849=0.0)
	(atposition counter#1::chewie_door_right_joint pstn18844=1.872)
	(atposition oven#1::knob_joint_2 pstn18847=0.0)

	(pose chicken-leg p47847=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(pose braiserlid#1 p47848=(0.567, 7.872, 0.712, 0.0, -0.0, 2.857))
	(pose salt-shaker p47850=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(pose pepper-shaker p47851=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose braiserbody#1 p47846=(0.7, 8.9, 0.923, 0.0, -0.0, 1.571))

	(atpose salt-shaker p47850=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(atpose chicken-leg p47847=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(atpose pepper-shaker p47851=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(atpose braiserlid#1 p47848=(0.567, 7.872, 0.712, 0.0, -0.0, 2.857))
	(atpose braiserbody#1 p47846=(0.7, 8.9, 0.923, 0.0, -0.0, 1.571))

	(aconf left aq848=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(aconf right aq136=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(isopenedposition counter#1::chewie_door_left_joint pstn18845=-1.872)
	(isopenedposition fridge#1::fridge_door pstn18846=1.78)
	(isopenedposition counter#1::chewie_door_right_joint pstn18844=1.872)

	(ataconf right aq136=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(ataconf left aq848=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(contained salt-shaker p47850=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained pepper-shaker p47851=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)

	(supported braiserbody#1 p47849=(0.7, 8.9, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)
	(supported braiserlid#1 p47848=(0.567, 7.872, 0.712, 0.0, -0.0, 2.857) counter#1::front_left_stove)

  )

  (:goal (and
    (open fridge#1::fridge_door)
  ))
)
        