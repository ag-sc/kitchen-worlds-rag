
(define
  (problem test_kitchen_dishwasher_260323_183233_seed_684592)
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

	(handempty right)
	(handempty left)

	(food chicken-leg)

	(controllable right)
	(controllable left)

	(sprinkler salt-shaker)
	(sprinkler pepper-shaker)

	(graspable braiserbody#1)
	(graspable salt-shaker)
	(graspable fork)
	(graspable braiserlid#1)
	(graspable pepper-shaker)
	(graspable chicken-leg)

	(space counter#1::sektion)
	(space braiserbody#1)
	(space counter#1::indigo_drawer_top)

	(joint oven#1::knob_joint_3)
	(joint counter#1::indigo_drawer_top_joint)
	(joint oven#1::knob_joint_2)
	(joint counter#1::chewie_door_left_joint)
	(joint faucet#1::joint_faucet_0)
	(joint counter#1::chewie_door_right_joint)
	(joint fridge#1::fridge_door)

	(region counter#1::indigo_tmp)
	(region fridge#1::shelf_top)
	(region basin#1::basin_bottom)
	(region counter#1::indigo_drawer_top)
	(region counter#1::front_left_stove)
	(region counter#1::sektion)
	(region braiserbody#1::braiser_bottom)
	(region counter#1::front_right_stove)
	(region braiserbody#1)
	(region counter#1::hitman_countertop)

	(staticlink basin#1::basin_bottom)
	(staticlink counter#1::front_left_stove)
	(staticlink counter#1::sektion)
	(staticlink counter#1::hitman_countertop)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink counter#1::front_right_stove)
	(staticlink braiserbody#1)
	(staticlink fridge#1::shelf_top)
	(staticlink counter#1::indigo_tmp)

	(bconf q648=(2.0, 6.25, 0.2, 3.142))

	(surface counter#1::front_left_stove)
	(surface braiserbody#1::braiser_bottom)
	(surface counter#1::front_right_stove)
	(surface braiserbody#1)
	(surface counter#1::hitman_countertop)
	(surface counter#1::indigo_tmp)
	(surface fridge#1::shelf_top)
	(surface basin#1::basin_bottom)

	(atbconf q648=(2.0, 6.25, 0.2, 3.142))
	(unattachedjoint oven#1::knob_joint_2)
	(unattachedjoint counter#1::chewie_door_left_joint)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint oven#1::knob_joint_3)

	(door counter#1::chewie_door_left_joint)
	(door fridge#1::fridge_door)
	(door counter#1::chewie_door_right_joint)

	(movablelink counter#1::indigo_drawer_top)

	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)
	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto counter#1::indigo_drawer_top_joint counter#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable fork counter#1::indigo_tmp)
	(stackable braiserlid#1 braiserbody#1)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable chicken-leg basin#1::basin_bottom)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable chicken-leg braiserbody#1::braiser_bottom)

	(atposition fridge#1::fridge_door pstn3591=0.0)
	(atposition oven#1::knob_joint_2 pstn3592=0.0)
	(atposition counter#1::indigo_drawer_top_joint pstn3588=0.0)
	(atposition counter#1::chewie_door_left_joint pstn3590=0.0)
	(atposition counter#1::chewie_door_right_joint pstn3589=0.0)
	(atposition oven#1::knob_joint_3 pstn3593=0.0)
	(atposition faucet#1::joint_faucet_0 pstn3594=0.0)

	(containable pepper-shaker counter#1::indigo_drawer_top)
	(containable chicken-leg counter#1::sektion)
	(containable braiserbody#1 counter#1::indigo_drawer_top)
	(containable braiserlid#1 counter#1::sektion)
	(containable chicken-leg braiserbody#1)
	(containable salt-shaker counter#1::indigo_drawer_top)
	(containable fork counter#1::indigo_drawer_top)
	(containable pepper-shaker counter#1::sektion)
	(containable braiserbody#1 counter#1::sektion)
	(containable chicken-leg counter#1::indigo_drawer_top)
	(containable braiserlid#1 counter#1::indigo_drawer_top)
	(containable salt-shaker counter#1::sektion)
	(containable fork counter#1::sektion)

	(position counter#1::indigo_drawer_top_joint pstn3588=0.0)
	(position counter#1::chewie_door_right_joint pstn3589=0.0)
	(position counter#1::chewie_door_left_joint pstn3590=0.0)
	(position fridge#1::fridge_door pstn3591=0.0)
	(position oven#1::knob_joint_2 pstn3592=0.0)
	(position oven#1::knob_joint_3 pstn3593=0.0)
	(position faucet#1::joint_faucet_0 pstn3594=0.0)

	(atpose chicken-leg p2652=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(atpose salt-shaker p2653=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(atpose braiserbody#1 p2649=(0.7, 8.741, 0.923, 0.0, -0.0, 1.571))
	(atpose counter#1::indigo_drawer_top lp100=(0.594, 8.843, 0.722, 0.0, -0.0, 0.0))
	(atpose pepper-shaker p2654=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(atpose braiserlid#1 p2650=(0.7, 8.741, 0.953, 0.0, -0.0, 1.571))
	(isclosedposition counter#1::chewie_door_right_joint pstn3589=0.0)
	(isclosedposition fridge#1::fridge_door pstn3591=0.0)
	(isclosedposition faucet#1::joint_faucet_0 pstn3594=0.0)
	(isclosedposition oven#1::knob_joint_2 pstn3592=0.0)
	(isclosedposition counter#1::indigo_drawer_top_joint pstn3588=0.0)
	(isclosedposition counter#1::chewie_door_left_joint pstn3590=0.0)
	(isclosedposition oven#1::knob_joint_3 pstn3593=0.0)

	(aconf left aq24=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(aconf right aq624=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))

	(ataconf left aq24=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(ataconf right aq624=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))

	(pose counter#1::indigo_drawer_top lp100=(0.594, 8.843, 0.722, 0.0, -0.0, 0.0))
	(pose braiserlid#1 p2650=(0.7, 8.741, 0.953, 0.0, -0.0, 1.571))
	(pose pepper-shaker p2654=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose salt-shaker p2653=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(pose braiserbody#1 p2649=(0.7, 8.741, 0.923, 0.0, -0.0, 1.571))
	(pose chicken-leg p2652=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))

	(jointaffectlink counter#1::indigo_drawer_top_joint counter#1::indigo_drawer_top)

	(startpose counter#1::indigo_drawer_top lp100=(0.594, 8.843, 0.722, 0.0, -0.0, 0.0))

	(relpose fork rp99=(0.141, -0.012, -0.033, 0.0, -0.0, 2.447) counter#1::indigo_drawer_top)

	(contained fork p2655=(0.735, 8.831, 0.689, 0.0, -0.0, 2.447) counter#1::indigo_drawer_top)
	(contained pepper-shaker p2654=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained salt-shaker p2653=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)
	(supported braiserbody#1 p2651=(0.7, 8.741, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)
	(supported braiserlid#1 p2650=(0.7, 8.741, 0.953, 0.0, -0.0, 1.571) braiserbody#1)

	(atrelpose fork rp99=(0.141, -0.012, -0.033, 0.0, -0.0, 2.447) counter#1::indigo_drawer_top)

  )

  (:goal (and
    (open fridge#1::fridge_door)
  ))
)
        