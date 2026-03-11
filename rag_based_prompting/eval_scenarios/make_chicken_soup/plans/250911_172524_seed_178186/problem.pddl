
(define
  (problem test_kitchen_chicken_soup_250911_172524_seed_178186)
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

	(canpull left)
	(canpull right)

	(cangrasphandle)
	(handempty left)
	(handempty right)

	(food chicken-leg)

	(controllable right)
	(controllable left)

	(space braiserbody#1)
	(space counter#1::sektion)

	(graspable chicken-leg)
	(graspable braiserbody#1)
	(graspable salt-shaker)
	(graspable braiserlid#1)
	(graspable pepper-shaker)
	(sprinkler salt-shaker)
	(sprinkler pepper-shaker)

	(door fridge#1::fridge_door)
	(door counter#1::chewie_door_left_joint)
	(door counter#1::chewie_door_right_joint)

	(surface counter#1::indigo_tmp)
	(surface counter#1::front_right_stove)
	(surface basin#1::basin_bottom)
	(surface counter#1::front_left_stove)
	(surface braiserbody#1::braiser_bottom)
	(surface braiserbody#1)
	(surface counter#1::hitman_countertop)
	(surface fridge#1::shelf_top)

	(bconf q672=(2.0, 6.25, 0.2, 3.142))
	(region counter#1::front_left_stove)
	(region basin#1::basin_bottom)
	(region braiserbody#1::braiser_bottom)
	(region braiserbody#1)
	(region counter#1::hitman_countertop)
	(region fridge#1::shelf_top)
	(region counter#1::sektion)
	(region counter#1::indigo_tmp)
	(region counter#1::front_right_stove)

	(atbconf q672=(2.0, 6.25, 0.2, 3.142))

	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)

	(staticlink counter#1::hitman_countertop)
	(staticlink fridge#1::shelf_top)
	(staticlink counter#1::sektion)
	(staticlink counter#1::indigo_tmp)
	(staticlink counter#1::front_right_stove)
	(staticlink basin#1::basin_bottom)
	(staticlink counter#1::front_left_stove)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink braiserbody#1)

	(joint counter#1::chewie_door_right_joint)
	(joint faucet#1::joint_faucet_0)
	(joint fridge#1::fridge_door)
	(joint oven#1::knob_joint_2)
	(joint oven#1::knob_joint_3)
	(joint counter#1::chewie_door_left_joint)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint oven#1::knob_joint_2)
	(unattachedjoint oven#1::knob_joint_3)
	(unattachedjoint counter#1::chewie_door_left_joint)
	(unattachedjoint counter#1::chewie_door_right_joint)

	(containable salt-shaker counter#1::sektion)
	(containable pepper-shaker counter#1::sektion)
	(containable chicken-leg braiserbody#1)
	(containable braiserlid#1 counter#1::sektion)
	(containable chicken-leg counter#1::sektion)
	(containable braiserbody#1 counter#1::sektion)

	(position oven#1::knob_joint_2 pstn15488=0.0)
	(position counter#1::chewie_door_right_joint pstn15485=1.872)
	(position counter#1::chewie_door_left_joint pstn15486=-1.872)
	(position oven#1::knob_joint_3 pstn15489=0.0)
	(position faucet#1::joint_faucet_0 pstn15490=0.0)
	(position fridge#1::fridge_door pstn15487=1.78)

	(isclosedposition oven#1::knob_joint_3 pstn15489=0.0)
	(isclosedposition faucet#1::joint_faucet_0 pstn15490=0.0)
	(isclosedposition oven#1::knob_joint_2 pstn15488=0.0)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable braiserlid#1 braiserbody#1)
	(stackable chicken-leg basin#1::basin_bottom)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable salt-shaker counter#1::indigo_tmp)

	(isopenedposition fridge#1::fridge_door pstn15487=1.78)
	(isopenedposition counter#1::chewie_door_left_joint pstn15486=-1.872)
	(isopenedposition counter#1::chewie_door_right_joint pstn15485=1.872)

	(atposition counter#1::chewie_door_left_joint pstn15486=-1.872)
	(atposition counter#1::chewie_door_right_joint pstn15485=1.872)
	(atposition oven#1::knob_joint_3 pstn15489=0.0)
	(atposition faucet#1::joint_faucet_0 pstn15490=0.0)
	(atposition fridge#1::fridge_door pstn15487=1.78)
	(atposition oven#1::knob_joint_2 pstn15488=0.0)

	(pose braiserlid#1 p23516=(0.567, 7.872, 0.712, 0.0, -0.0, 3.036))
	(pose braiserbody#1 p23514=(0.7, 8.769, 0.923, 0.0, -0.0, 1.571))
	(pose salt-shaker p23518=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(pose pepper-shaker p23519=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose chicken-leg p23515=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))

	(atpose braiserlid#1 p23516=(0.567, 7.872, 0.712, 0.0, -0.0, 3.036))
	(atpose braiserbody#1 p23514=(0.7, 8.769, 0.923, 0.0, -0.0, 1.571))
	(atpose pepper-shaker p23519=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(atpose chicken-leg p23515=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(atpose salt-shaker p23518=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))

	(aconf left aq896=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(aconf right aq712=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))

	(ataconf right aq712=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(ataconf left aq896=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(contained salt-shaker p23518=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained pepper-shaker p23519=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)

	(supported braiserbody#1 p23517=(0.7, 8.769, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)
	(supported braiserlid#1 p23516=(0.567, 7.872, 0.712, 0.0, -0.0, 3.036) counter#1::front_left_stove)

  )

  (:goal (and
    (pick chicken-leg)
  ))
)
        