/**
 * File: spout_writer.js
 * Project: Jovi_Spout
 *
 */

import { app } from "../../../scripts/app.js"
import { widgetSizeModeHook } from '../util/util_jov.js'

const _id = "SPOUT WRITER (JOV_SP) 🎥"

app.registerExtension({
	name: 'Jovi_Spout.node.' + _id,
	async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== _id) {
            return;
        }

        // widgetSizeModeHook(nodeType);
	}
})
