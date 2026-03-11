
(define
  (problem test_kitchen_chicken_soup_250902_121044_seed_318962)
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
	(surface braiserbody#1)
	(surface counter#1::hitman_countertop)
	(surface fridge#1::shelf_top)
	(surface basin#1::basin_bottom)
	(surface counter#1::front_left_stove)
	(surface braiserbody#1::braiser_bottom)
	(surface counter#1::indigo_tmp)
	(surface counter#1::front_right_stove)

	(graspable braiserlid#1)
	(graspable pepper-shaker)
	(graspable chicken-leg)
	(graspable salt-shaker)
	(graspable braiserbody#1)

	(space counter#1::sektion)
	(space braiserbody#1)

	(region basin#1::basin_bottom)
	(region counter#1::front_left_stove)
	(region counter#1::sektion)
	(region braiserbody#1::braiser_bottom)
	(region counter#1::indigo_tmp)
	(region counter#1::front_right_stove)
	(region braiserbody#1)
	(region counter#1::hitman_countertop)
	(region fridge#1::shelf_top)

	(bconf q48=(2.0, 6.25, 0.2, 3.142))

	(atbconf q48=(2.0, 6.25, 0.2, 3.142))

	(containable chicken-leg braiserbody#1)
	(containable salt-shaker counter#1::sektion)
	(containable pepper-shaker counter#1::sektion)
	(containable braiserlid#1 counter#1::sektion)
	(containable chicken-leg counter#1::sektion)
	(containable braiserbody#1 counter#1::sektion)

	(door counter#1::chewie_door_left_joint)
	(door counter#1::chewie_door_right_joint)
	(door fridge#1::fridge_door)

	(staticlink counter#1::front_right_stove)
	(staticlink counter#1::sektion)
	(staticlink braiserbody#1)
	(staticlink counter#1::hitman_countertop)
	(staticlink fridge#1::shelf_top)
	(staticlink basin#1::basin_bottom)
	(staticlink counter#1::front_left_stove)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink counter#1::indigo_tmp)

	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)
	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(joint counter#1::chewie_door_right_joint)
	(joint faucet#1::joint_faucet_0)
	(joint fridge#1::fridge_door)
	(joint oven#1::knob_joint_2)
	(joint oven#1::knob_joint_3)
	(joint counter#1::chewie_door_left_joint)

	(position faucet#1::joint_faucet_0 pstn7718=0.0)
	(position counter#1::chewie_door_left_joint pstn7714=-1.872)
	(position fridge#1::fridge_door pstn7715=1.78)
	(position counter#1::chewie_door_right_joint pstn7713=1.872)
	(position oven#1::knob_joint_2 pstn7716=0.0)
	(position oven#1::knob_joint_3 pstn7717=0.0)

	(isclosedposition oven#1::knob_joint_3 pstn7717=0.0)
	(isclosedposition oven#1::knob_joint_2 pstn7716=0.0)
	(isclosedposition faucet#1::joint_faucet_0 pstn7718=0.0)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable chicken-leg basin#1::basin_bottom)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 braiserbody#1)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint oven#1::knob_joint_2)
	(unattachedjoint oven#1::knob_joint_3)
	(unattachedjoint counter#1::chewie_door_left_joint)

	(atposition counter#1::chewie_door_left_joint pstn7714=-1.872)
	(atposition fridge#1::fridge_door pstn7715=1.78)
	(atposition counter#1::chewie_door_right_joint pstn7713=1.872)
	(atposition oven#1::knob_joint_3 pstn7717=0.0)
	(atposition oven#1::knob_joint_2 pstn7716=0.0)
	(atposition faucet#1::joint_faucet_0 pstn7718=0.0)

	(pose braiserlid#1 p8715=(0.567, 7.872, 0.712, 0.0, -0.0, 0.79))
	(pose salt-shaker p8717=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(pose braiserbody#1 p8713=(0.7, 8.851, 0.923, 0.0, -0.0, 1.571))
	(pose pepper-shaker p8718=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose chicken-leg p8714=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))

	(atpose braiserlid#1 p8715=(0.567, 7.872, 0.712, 0.0, -0.0, 0.79))
	(atpose salt-shaker p8717=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(atpose pepper-shaker p8718=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(atpose braiserbody#1 p8713=(0.7, 8.851, 0.923, 0.0, -0.0, 1.571))
	(atpose chicken-leg p8714=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))

	(isopenedposition counter#1::chewie_door_right_joint pstn7713=1.872)
	(isopenedposition counter#1::chewie_door_left_joint pstn7714=-1.872)
	(isopenedposition fridge#1::fridge_door pstn7715=1.78)

	(ataconf left aq312=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(ataconf right aq576=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))

	(aconf right aq576=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(aconf left aq312=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(contained salt-shaker p8717=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained pepper-shaker p8718=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)

	(supported braiserbody#1 p8716=(0.7, 8.851, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)
	(supported braiserlid#1 p8715=(0.567, 7.872, 0.712, 0.0, -0.0, 0.79) counter#1::front_left_stove)

  )

  (:goal (and
    (open fridge#1::fridge_door)
  ))
)
        