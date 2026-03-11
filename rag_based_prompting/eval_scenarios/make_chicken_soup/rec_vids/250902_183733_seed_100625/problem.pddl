
(define
  (problem test_kitchen_chicken_soup_250902_183733_seed_100625)
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

	(graspable chicken-leg)
	(graspable salt-shaker)
	(graspable braiserbody#1)
	(graspable braiserlid#1)
	(graspable pepper-shaker)
	(sprinkler salt-shaker)
	(sprinkler pepper-shaker)

	(space counter#1::sektion)
	(space braiserbody#1)

	(joint oven#1::knob_joint_3)
	(joint counter#1::chewie_door_left_joint)
	(joint counter#1::chewie_door_right_joint)
	(joint faucet#1::joint_faucet_0)
	(joint fridge#1::fridge_door)
	(joint oven#1::knob_joint_2)
	(region fridge#1::shelf_top)
	(region counter#1::sektion)
	(region basin#1::basin_bottom)
	(region braiserbody#1::braiser_bottom)
	(region braiserbody#1)
	(region counter#1::front_right_stove)
	(region counter#1::hitman_countertop)
	(region counter#1::front_left_stove)
	(region counter#1::indigo_tmp)

	(bconf q216=(2.0, 6.25, 0.2, 3.142))

	(atbconf q216=(2.0, 6.25, 0.2, 3.142))
	(unattachedjoint oven#1::knob_joint_3)
	(unattachedjoint oven#1::knob_joint_2)
	(unattachedjoint counter#1::chewie_door_left_joint)
	(unattachedjoint counter#1::chewie_door_right_joint)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint fridge#1::fridge_door)

	(surface braiserbody#1::braiser_bottom)
	(surface counter#1::indigo_tmp)
	(surface braiserbody#1)
	(surface counter#1::front_right_stove)
	(surface counter#1::hitman_countertop)
	(surface counter#1::front_left_stove)
	(surface fridge#1::shelf_top)
	(surface basin#1::basin_bottom)

	(door counter#1::chewie_door_left_joint)
	(door counter#1::chewie_door_right_joint)
	(door fridge#1::fridge_door)

	(staticlink braiserbody#1::braiser_bottom)
	(staticlink counter#1::indigo_tmp)
	(staticlink braiserbody#1)
	(staticlink counter#1::front_right_stove)
	(staticlink counter#1::hitman_countertop)
	(staticlink counter#1::front_left_stove)
	(staticlink fridge#1::shelf_top)
	(staticlink counter#1::sektion)
	(staticlink basin#1::basin_bottom)

	(containable chicken-leg counter#1::sektion)
	(containable braiserbody#1 counter#1::sektion)
	(containable salt-shaker counter#1::sektion)
	(containable braiserlid#1 counter#1::sektion)
	(containable pepper-shaker counter#1::sektion)
	(containable chicken-leg braiserbody#1)

	(stackable chicken-leg basin#1::basin_bottom)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::hitman_countertop)
	(stackable braiserlid#1 braiserbody#1)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable braiserlid#1 counter#1::indigo_tmp)

	(atposition oven#1::knob_joint_3 pstn17474=0.0)
	(atposition counter#1::chewie_door_right_joint pstn17470=1.872)
	(atposition fridge#1::fridge_door pstn17472=1.78)
	(atposition oven#1::knob_joint_2 pstn17473=0.0)
	(atposition counter#1::chewie_door_left_joint pstn17471=-1.872)
	(atposition faucet#1::joint_faucet_0 pstn17475=0.0)

	(position faucet#1::joint_faucet_0 pstn17475=0.0)
	(position fridge#1::fridge_door pstn17472=1.78)
	(position counter#1::chewie_door_right_joint pstn17470=1.872)
	(position oven#1::knob_joint_3 pstn17474=0.0)
	(position oven#1::knob_joint_2 pstn17473=0.0)
	(position counter#1::chewie_door_left_joint pstn17471=-1.872)

	(isjointto counter#1::chewie_door_left_joint counter#1)
	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)
	(isopenedposition fridge#1::fridge_door pstn17472=1.78)
	(isopenedposition counter#1::chewie_door_right_joint pstn17470=1.872)
	(isopenedposition counter#1::chewie_door_left_joint pstn17471=-1.872)

	(isclosedposition faucet#1::joint_faucet_0 pstn17475=0.0)
	(isclosedposition oven#1::knob_joint_3 pstn17474=0.0)
	(isclosedposition oven#1::knob_joint_2 pstn17473=0.0)

	(pose pepper-shaker p39846=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose chicken-leg p39842=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(pose braiserlid#1 p39843=(0.567, 7.872, 0.712, 0.0, -0.0, 2.975))
	(pose salt-shaker p39845=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(pose braiserbody#1 p39841=(0.7, 8.959, 0.923, 0.0, -0.0, 1.571))

	(aconf left aq968=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))
	(aconf right aq880=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(atpose pepper-shaker p39846=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(atpose braiserlid#1 p39843=(0.567, 7.872, 0.712, 0.0, -0.0, 2.975))
	(atpose braiserbody#1 p39841=(0.7, 8.959, 0.923, 0.0, -0.0, 1.571))
	(atpose chicken-leg p39842=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(atpose salt-shaker p39845=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))

	(ataconf right aq880=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(ataconf left aq968=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(contained pepper-shaker p39846=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained salt-shaker p39845=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)

	(supported braiserbody#1 p39844=(0.7, 8.959, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)
	(supported braiserlid#1 p39843=(0.567, 7.872, 0.712, 0.0, -0.0, 2.975) counter#1::front_left_stove)

  )

  (:goal (and
    (open fridge#1::fridge_door)
  ))
)
        