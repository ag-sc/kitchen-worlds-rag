
(define
  (problem test_kitchen_chicken_soup_250903_103225_seed_988376)
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

	(graspable chicken-leg)
	(graspable salt-shaker)
	(graspable braiserbody#1)
	(graspable braiserlid#1)
	(graspable pepper-shaker)
	(sprinkler salt-shaker)
	(sprinkler pepper-shaker)

	(space counter#1::sektion)
	(space braiserbody#1)

	(region counter#1::indigo_tmp)
	(region basin#1::basin_bottom)
	(region counter#1::front_right_stove)
	(region counter#1::sektion)
	(region counter#1::hitman_countertop)
	(region braiserbody#1::braiser_bottom)
	(region braiserbody#1)
	(region fridge#1::shelf_top)
	(region counter#1::front_left_stove)

	(surface basin#1::basin_bottom)
	(surface counter#1::hitman_countertop)
	(surface braiserbody#1::braiser_bottom)
	(surface counter#1::indigo_tmp)
	(surface counter#1::front_right_stove)
	(surface fridge#1::shelf_top)
	(surface counter#1::front_left_stove)
	(surface braiserbody#1)

	(staticlink counter#1::indigo_tmp)
	(staticlink basin#1::basin_bottom)
	(staticlink counter#1::front_right_stove)
	(staticlink counter#1::sektion)
	(staticlink counter#1::hitman_countertop)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink braiserbody#1)
	(staticlink fridge#1::shelf_top)
	(staticlink counter#1::front_left_stove)

	(bconf q528=(2.0, 6.25, 0.2, 3.142))

	(atbconf q528=(2.0, 6.25, 0.2, 3.142))

	(door counter#1::chewie_door_left_joint)
	(door counter#1::chewie_door_right_joint)
	(door fridge#1::fridge_door)

	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)
	(joint counter#1::chewie_door_right_joint)
	(joint faucet#1::joint_faucet_0)
	(joint fridge#1::fridge_door)
	(joint oven#1::knob_joint_3)
	(joint oven#1::knob_joint_2)
	(joint counter#1::chewie_door_left_joint)

	(containable braiserlid#1 counter#1::sektion)
	(containable braiserbody#1 counter#1::sektion)
	(containable chicken-leg counter#1::sektion)
	(containable salt-shaker counter#1::sektion)
	(containable chicken-leg braiserbody#1)
	(containable pepper-shaker counter#1::sektion)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable chicken-leg basin#1::basin_bottom)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable braiserlid#1 braiserbody#1)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable braiserbody#1 counter#1::indigo_tmp)

	(atposition fridge#1::fridge_door pstn36490=1.78)
	(atposition counter#1::chewie_door_right_joint pstn36488=1.872)
	(atposition oven#1::knob_joint_2 pstn36491=0.0)
	(atposition faucet#1::joint_faucet_0 pstn36493=0.0)
	(atposition counter#1::chewie_door_left_joint pstn36489=-1.872)
	(atposition oven#1::knob_joint_3 pstn36492=0.0)

	(unattachedjoint counter#1::chewie_door_left_joint)
	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint oven#1::knob_joint_3)
	(unattachedjoint oven#1::knob_joint_2)

	(isclosedposition oven#1::knob_joint_2 pstn36491=0.0)
	(isclosedposition faucet#1::joint_faucet_0 pstn36493=0.0)
	(isclosedposition oven#1::knob_joint_3 pstn36492=0.0)

	(position counter#1::chewie_door_right_joint pstn36488=1.872)
	(position oven#1::knob_joint_2 pstn36491=0.0)
	(position faucet#1::joint_faucet_0 pstn36493=0.0)
	(position counter#1::chewie_door_left_joint pstn36489=-1.872)
	(position oven#1::knob_joint_3 pstn36492=0.0)
	(position fridge#1::fridge_door pstn36490=1.78)

	(pose pepper-shaker p112700=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose braiserlid#1 p112697=(0.567, 7.872, 0.712, 0.0, -0.0, 1.975))
	(pose braiserbody#1 p112695=(0.7, 8.844, 0.923, 0.0, -0.0, 1.571))
	(pose chicken-leg p112696=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(pose salt-shaker p112699=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))

	(isopenedposition counter#1::chewie_door_left_joint pstn36489=-1.872)
	(isopenedposition fridge#1::fridge_door pstn36490=1.78)
	(isopenedposition counter#1::chewie_door_right_joint pstn36488=1.872)

	(atpose pepper-shaker p112700=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(atpose braiserlid#1 p112697=(0.567, 7.872, 0.712, 0.0, -0.0, 1.975))
	(atpose braiserbody#1 p112695=(0.7, 8.844, 0.923, 0.0, -0.0, 1.571))
	(atpose salt-shaker p112699=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(atpose chicken-leg p112696=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))

	(aconf right aq768=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(aconf left aq128=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(ataconf right aq768=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(ataconf left aq128=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(contained pepper-shaker p112700=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained salt-shaker p112699=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)

	(supported braiserbody#1 p112698=(0.7, 8.844, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)
	(supported braiserlid#1 p112697=(0.567, 7.872, 0.712, 0.0, -0.0, 1.975) counter#1::front_left_stove)

  )

  (:goal (and
    (open fridge#1::fridge_door)
  ))
)
        