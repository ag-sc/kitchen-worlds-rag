
(define
  (problem test_kitchen_chicken_soup_250829_002141_seed_537415)
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

	(arm right)
	(arm left)

	(canmovebase)

	(canpull left)
	(canpull right)

	(cangrasphandle)

	(handempty right)
	(handempty left)

	(food chicken-leg)

	(controllable right)
	(controllable left)

	(space braiserbody#1)
	(space counter#1::sektion)

	(graspable chicken-leg)
	(graspable salt-shaker)
	(graspable braiserbody#1)
	(graspable braiserlid#1)
	(graspable pepper-shaker)

	(sprinkler pepper-shaker)
	(sprinkler salt-shaker)

	(region fridge#1::shelf_top)
	(region counter#1::sektion)
	(region basin#1::basin_bottom)
	(region braiserbody#1)
	(region counter#1::hitman_countertop)
	(region counter#1::front_left_stove)
	(region braiserbody#1::braiser_bottom)
	(region counter#1::indigo_tmp)
	(region counter#1::front_right_stove)

	(staticlink counter#1::sektion)
	(staticlink basin#1::basin_bottom)
	(staticlink braiserbody#1)
	(staticlink counter#1::hitman_countertop)
	(staticlink counter#1::front_left_stove)
	(staticlink counter#1::front_right_stove)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink counter#1::indigo_tmp)
	(staticlink fridge#1::shelf_top)

	(bconf q584=(2.0, 6.25, 0.2, 3.142))

	(atbconf q584=(2.0, 6.25, 0.2, 3.142))

	(surface braiserbody#1::braiser_bottom)
	(surface counter#1::indigo_tmp)
	(surface counter#1::front_right_stove)
	(surface fridge#1::shelf_top)
	(surface basin#1::basin_bottom)
	(surface braiserbody#1)
	(surface counter#1::hitman_countertop)
	(surface counter#1::front_left_stove)

	(door counter#1::chewie_door_right_joint)
	(door fridge#1::fridge_door)
	(door counter#1::chewie_door_left_joint)
	(joint counter#1::chewie_door_left_joint)
	(joint counter#1::chewie_door_right_joint)
	(joint oven#1::knob_joint_3)
	(joint faucet#1::joint_faucet_0)
	(joint fridge#1::fridge_door)
	(joint oven#1::knob_joint_2)

	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)
	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto oven#1::knob_joint_2 oven#1)

	(containable braiserbody#1 counter#1::sektion)
	(containable pepper-shaker counter#1::sektion)
	(containable chicken-leg counter#1::sektion)
	(containable salt-shaker counter#1::sektion)
	(containable chicken-leg braiserbody#1)
	(containable braiserlid#1 counter#1::sektion)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable braiserlid#1 braiserbody#1)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable chicken-leg basin#1::basin_bottom)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable pepper-shaker counter#1::indigo_tmp)

	(position faucet#1::joint_faucet_0 pstn8182=0.0)
	(position counter#1::chewie_door_left_joint pstn8178=-1.872)
	(position oven#1::knob_joint_2 pstn8180=0.0)
	(position oven#1::knob_joint_3 pstn8181=0.0)
	(position fridge#1::fridge_door pstn8179=1.78)
	(position counter#1::chewie_door_right_joint pstn8177=1.872)

	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint oven#1::knob_joint_2)
	(unattachedjoint oven#1::knob_joint_3)
	(unattachedjoint counter#1::chewie_door_left_joint)

	(isclosedposition faucet#1::joint_faucet_0 pstn8182=0.0)
	(isclosedposition oven#1::knob_joint_3 pstn8181=0.0)
	(isclosedposition oven#1::knob_joint_2 pstn8180=0.0)

	(atposition counter#1::chewie_door_left_joint pstn8178=-1.872)
	(atposition faucet#1::joint_faucet_0 pstn8182=0.0)
	(atposition oven#1::knob_joint_2 pstn8180=0.0)
	(atposition oven#1::knob_joint_3 pstn8181=0.0)
	(atposition fridge#1::fridge_door pstn8179=1.78)
	(atposition counter#1::chewie_door_right_joint pstn8177=1.872)

	(pose chicken-leg p25522=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(pose braiserbody#1 p25521=(0.7, 8.727, 0.923, 0.0, -0.0, 1.571))
	(pose pepper-shaker p25526=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose salt-shaker p25525=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(pose braiserlid#1 p25523=(0.567, 7.872, 0.712, 0.0, -0.0, 2.718))

	(atpose salt-shaker p25525=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(atpose braiserlid#1 p25523=(0.567, 7.872, 0.712, 0.0, -0.0, 2.718))
	(atpose chicken-leg p25522=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(atpose braiserbody#1 p25521=(0.7, 8.727, 0.923, 0.0, -0.0, 1.571))
	(atpose pepper-shaker p25526=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))

	(isopenedposition counter#1::chewie_door_left_joint pstn8178=-1.872)
	(isopenedposition fridge#1::fridge_door pstn8179=1.78)
	(isopenedposition counter#1::chewie_door_right_joint pstn8177=1.872)

	(ataconf left aq368=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(ataconf right aq296=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))

	(aconf right aq296=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(aconf left aq368=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(contained salt-shaker p25525=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained pepper-shaker p25526=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)

	(supported braiserlid#1 p25523=(0.567, 7.872, 0.712, 0.0, -0.0, 2.718) counter#1::front_left_stove)
	(supported braiserbody#1 p25524=(0.7, 8.727, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)

  )

  (:goal (and
    (open fridge#1::fridge_door)
  ))
)
        