
(define
  (problem test_kitchen_chicken_soup_250910_120358_seed_854983)
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
	(handempty left)
	(handempty right)

	(food chicken-leg)

	(controllable left)
	(controllable right)

	(space braiserbody#1)
	(space counter#1::sektion)

	(sprinkler salt-shaker)
	(sprinkler pepper-shaker)

	(graspable braiserlid#1)
	(graspable pepper-shaker)
	(graspable chicken-leg)
	(graspable salt-shaker)
	(graspable braiserbody#1)

	(bconf q728=(2.0, 6.25, 0.2, 3.142))
	(region counter#1::front_left_stove)
	(region braiserbody#1::braiser_bottom)
	(region braiserbody#1)
	(region counter#1::front_right_stove)
	(region counter#1::hitman_countertop)
	(region counter#1::sektion)
	(region fridge#1::shelf_top)
	(region counter#1::indigo_tmp)
	(region basin#1::basin_bottom)

	(atbconf q728=(2.0, 6.25, 0.2, 3.142))
	(unattachedjoint oven#1::knob_joint_3)
	(unattachedjoint faucet#1::joint_faucet_0)
	(unattachedjoint fridge#1::fridge_door)
	(unattachedjoint oven#1::knob_joint_2)
	(unattachedjoint counter#1::chewie_door_left_joint)
	(unattachedjoint counter#1::chewie_door_right_joint)

	(isjointto oven#1::knob_joint_2 oven#1)
	(isjointto fridge#1::fridge_door fridge#1)
	(isjointto faucet#1::joint_faucet_0 faucet#1)
	(isjointto oven#1::knob_joint_3 oven#1)
	(isjointto counter#1::chewie_door_right_joint counter#1)
	(isjointto counter#1::chewie_door_left_joint counter#1)
	(surface braiserbody#1::braiser_bottom)
	(surface braiserbody#1)
	(surface counter#1::front_right_stove)
	(surface counter#1::hitman_countertop)
	(surface fridge#1::shelf_top)
	(surface counter#1::indigo_tmp)
	(surface basin#1::basin_bottom)
	(surface counter#1::front_left_stove)

	(staticlink counter#1::front_left_stove)
	(staticlink braiserbody#1::braiser_bottom)
	(staticlink braiserbody#1)
	(staticlink counter#1::front_right_stove)
	(staticlink counter#1::hitman_countertop)
	(staticlink counter#1::sektion)
	(staticlink fridge#1::shelf_top)
	(staticlink counter#1::indigo_tmp)
	(staticlink basin#1::basin_bottom)

	(door counter#1::chewie_door_right_joint)
	(door fridge#1::fridge_door)
	(door counter#1::chewie_door_left_joint)

	(joint counter#1::chewie_door_right_joint)
	(joint oven#1::knob_joint_3)
	(joint faucet#1::joint_faucet_0)
	(joint fridge#1::fridge_door)
	(joint oven#1::knob_joint_2)
	(joint counter#1::chewie_door_left_joint)

	(containable chicken-leg counter#1::sektion)
	(containable braiserbody#1 counter#1::sektion)
	(containable salt-shaker counter#1::sektion)
	(containable braiserlid#1 counter#1::sektion)
	(containable pepper-shaker counter#1::sektion)
	(containable chicken-leg braiserbody#1)

	(atposition oven#1::knob_joint_3 pstn5495=0.0)
	(atposition counter#1::chewie_door_right_joint pstn5491=1.872)
	(atposition faucet#1::joint_faucet_0 pstn5496=0.0)
	(atposition counter#1::chewie_door_left_joint pstn5492=-1.872)
	(atposition fridge#1::fridge_door pstn5493=1.78)
	(atposition oven#1::knob_joint_2 pstn5494=0.0)
	(position fridge#1::fridge_door pstn5493=1.78)
	(position oven#1::knob_joint_3 pstn5495=0.0)
	(position faucet#1::joint_faucet_0 pstn5496=0.0)
	(position counter#1::chewie_door_left_joint pstn5492=-1.872)
	(position oven#1::knob_joint_2 pstn5494=0.0)
	(position counter#1::chewie_door_right_joint pstn5491=1.872)

	(isclosedposition oven#1::knob_joint_3 pstn5495=0.0)
	(isclosedposition faucet#1::joint_faucet_0 pstn5496=0.0)
	(isclosedposition oven#1::knob_joint_2 pstn5494=0.0)
	(stackable braiserlid#1 counter#1::front_left_stove)
	(stackable chicken-leg counter#1::indigo_tmp)
	(stackable salt-shaker counter#1::indigo_tmp)
	(stackable braiserbody#1 counter#1::indigo_tmp)
	(stackable braiserlid#1 braiserbody#1)
	(stackable braiserlid#1 counter#1::hitman_countertop)
	(stackable chicken-leg basin#1::basin_bottom)
	(stackable braiserlid#1 counter#1::indigo_tmp)
	(stackable pepper-shaker counter#1::hitman_countertop)
	(stackable pepper-shaker counter#1::indigo_tmp)
	(stackable chicken-leg braiserbody#1::braiser_bottom)
	(stackable braiserlid#1 basin#1::basin_bottom)
	(stackable salt-shaker counter#1::hitman_countertop)

	(pose pepper-shaker p6074=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(pose salt-shaker p6073=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(pose braiserlid#1 p6071=(0.567, 7.872, 0.712, 0.0, -0.0, 1.303))
	(pose braiserbody#1 p6069=(0.7, 8.9, 0.923, 0.0, -0.0, 1.571))
	(pose chicken-leg p6070=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))

	(atpose pepper-shaker p6074=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142))
	(atpose braiserbody#1 p6069=(0.7, 8.9, 0.923, 0.0, -0.0, 1.571))
	(atpose braiserlid#1 p6071=(0.567, 7.872, 0.712, 0.0, -0.0, 1.303))
	(atpose chicken-leg p6070=(0.654, 4.846, 1.384, 0.0, 0.0, -0.366))
	(atpose salt-shaker p6073=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142))
	(isopenedposition counter#1::chewie_door_left_joint pstn5492=-1.872)
	(isopenedposition counter#1::chewie_door_right_joint pstn5491=1.872)
	(isopenedposition fridge#1::fridge_door pstn5493=1.78)

	(aconf right aq840=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(aconf left aq272=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(ataconf right aq840=(-0.677, -0.343, -1.2, -1.467, -1.242, -1.954, -2.223))
	(ataconf left aq272=(0.677, -0.343, 1.2, -1.467, 1.242, -1.954, 2.223))

	(supported braiserbody#1 p6072=(0.7, 8.9, 0.923, 0.0, -0.0, 1.571) counter#1::indigo_tmp)
	(supported braiserlid#1 p6071=(0.567, 7.872, 0.712, 0.0, -0.0, 1.303) counter#1::front_left_stove)

	(contained pepper-shaker p6074=(0.764, 7.303, 1.164, 0.0, -0.0, 3.142) counter#1::sektion)
	(contained salt-shaker p6073=(0.771, 7.071, 1.152, 0.0, -0.0, 3.142) counter#1::sektion)

  )

  (:goal (and
    (open fridge#1::fridge_door)
  ))
)
        