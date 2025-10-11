use crate::cna::{NpuCnaDesc, NpuCoreDesc};
use crate::dpu::NpuDpuDesc;
use crate::hw::*;

#[cfg(not(feature = "no_std"))]
use std::slice;
#[cfg(feature = "no_std")]
use core::slice;

/// Matrix multiplication parameters
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct MatmulParams {
    pub m: u16,
    pub k: u16,
    pub n: u16,
    pub input_dma: u32,
    pub weights_dma: u32,
    pub output_dma: u32,
    pub tasks: *mut u64,
    pub fp32tofp16: u8,
}

/// Generate matrix multiplication task
///
/// We're only using cna & core, dpu outputs to memory
fn gen_matmul_task(
    ops: &mut [u64],
    cna_desc: &NpuCnaDesc,
    core_desc: &NpuCoreDesc,
    dpu_desc: &NpuDpuDesc,
) {
    ops[0] = npuop(OP_REG_DPU, 0xE, DPU_S_POINTER);

    let mut value: u32 = ((cna_desc.proc_precision as u32 & 0x7) << 7)
        | ((cna_desc.in_precision as u32 & 0x7) << 4)
        | (cna_desc.conv_mode as u32 & 0xf);
    ops[1] = npuop(OP_REG_CNA, value, CNA_CONV_CON1);

    value = ((cna_desc.kernel_groups as u32 & 0xFF) << 16)
        | ((cna_desc.feature_grains as u32 & 0x3FF) << 4);
    ops[2] = npuop(OP_REG_CNA, value, CNA_CONV_CON2);

    value = ((cna_desc.conv_y_stride as u32 & 0x7) << 3) | (cna_desc.conv_x_stride as u32 & 0x7);
    ops[3] = npuop(OP_REG_CNA, value, CNA_CONV_CON3);

    value =
        ((cna_desc.datain_width as u32 & 0x7FF) << 16) | (cna_desc.datain_height as u32 & 0x7FF);
    ops[4] = npuop(OP_REG_CNA, value, CNA_DATA_SIZE0);

    value = ((cna_desc.datain_channel.wrapping_sub(1) as u32 & 0xFFFF) << 16)
        | (cna_desc.datain_channel as u32 & 0xFFFF);
    ops[5] = npuop(OP_REG_CNA, value, CNA_DATA_SIZE1);

    value = cna_desc.dataout_width as u32 & 0x7FF;
    ops[6] = npuop(OP_REG_CNA, value, CNA_DATA_SIZE2);

    value = cna_desc.dataout_atomics & 0x3FFFF;
    ops[7] = npuop(OP_REG_CNA, value, CNA_DATA_SIZE3);

    value = cna_desc.weight_bytes;
    ops[8] = npuop(OP_REG_CNA, value, CNA_WEIGHT_SIZE0);

    value = cna_desc.weight_bytes_per_kernel & 0x7FFFF;
    ops[9] = npuop(OP_REG_CNA, value, CNA_WEIGHT_SIZE1);

    value = ((cna_desc.weight_width as u32 & 0x1F) << 24)
        | ((cna_desc.weight_height as u32 & 0x1F) << 16)
        | (cna_desc.weight_kernels as u32 & 0x3FFF);
    ops[10] = npuop(OP_REG_CNA, value, CNA_WEIGHT_SIZE2);

    value = ((cna_desc.weight_bank as u32 & 0xF) << 4) | (cna_desc.data_bank as u32 & 0xF);
    ops[11] = npuop(OP_REG_CNA, value, CNA_CBUF_CON0);

    value = cna_desc.data_entries as u32 & 0x1FFF;
    ops[12] = npuop(OP_REG_CNA, value, CNA_CBUF_CON1);

    value = ((cna_desc.data_sign as u32 & 0x1) << 3)
        | ((cna_desc.cvt_type as u32 & 0x1) << 1)
        | (cna_desc.cvt_bypass as u32 & 0x1);
    ops[13] = npuop(OP_REG_CNA, value, CNA_CVT_CON0);

    value = ((cna_desc.cvt_scale0 as u32 & 0xFFFF) << 16) | 0x0;
    ops[14] = npuop(OP_REG_CNA, value, CNA_CVT_CON1);

    value = ((cna_desc.cvt_scale1 as u32 & 0xFFFF) << 16) | 0x0;
    ops[15] = npuop(OP_REG_CNA, value, CNA_CVT_CON2);

    value = ((cna_desc.cvt_scale2 as u32 & 0xFFFF) << 16) | 0x0;
    ops[16] = npuop(OP_REG_CNA, value, CNA_CVT_CON3);

    value = ((cna_desc.cvt_scale3 as u32 & 0xFFFF) << 16) | 0x0;
    ops[17] = npuop(OP_REG_CNA, value, CNA_CVT_CON4);

    value = cna_desc.fc_skip_en as u32 & 0x1;
    ops[18] = npuop(OP_REG_CNA, value, CNA_FC_CON0);

    value = cna_desc.data_offset as u32 & 0x1FFFF;
    ops[19] = npuop(OP_REG_CNA, value, CNA_FC_CON1);

    value = ((cna_desc.pad_left as u32 & 0xF) << 4) | (cna_desc.pad_top as u32 & 0xF);
    ops[20] = npuop(OP_REG_CNA, value, CNA_PAD_CON0);

    ops[21] = npuop(
        OP_REG_CNA,
        cna_desc.feature_base_addr,
        CNA_FEATURE_DATA_ADDR,
    );

    value = cna_desc.weight_offset as u32 & 0x1FFFF;
    ops[22] = npuop(OP_REG_CNA, value, CNA_FC_CON2);

    value =
        ((cna_desc.weight_burst_len as u32 & 0xF) << 16) | (cna_desc.data_burst_len as u32 & 0xF);
    ops[23] = npuop(OP_REG_CNA, value, CNA_DMA_CON0);

    value = cna_desc.line_stride & 0xFFFFFFF;
    ops[24] = npuop(OP_REG_CNA, value, CNA_DMA_CON1);

    value = cna_desc.surf_stride as u32 & 0xFFFFFFF;
    ops[25] = npuop(OP_REG_CNA, value, CNA_DMA_CON2);

    value = ((cna_desc.dma_width as u32 & 0x7FF) << 16) | (cna_desc.dma_height as u32 & 0x7FF);
    ops[26] = npuop(OP_REG_CNA, value, CNA_FC_DATA_SIZE0);

    value = cna_desc.dma_channel as u32 & 0xFFFF;
    ops[27] = npuop(OP_REG_CNA, value, CNA_FC_DATA_SIZE1);

    ops[28] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_CTRL);
    ops[29] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_REGNUM);
    ops[30] = npuop(OP_REG_CNA, cna_desc.decompress_addr0, CNA_DCOMP_ADDR0);
    ops[31] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT);
    ops[32] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT1);
    ops[33] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT2);
    ops[34] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT3);
    ops[35] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT4);
    ops[36] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT5);
    ops[37] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT6);
    ops[38] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT7);
    ops[39] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT8);
    ops[40] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT9);
    ops[41] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT10);
    ops[42] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT11);
    ops[43] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT12);
    ops[44] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT13);
    ops[45] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT14);
    ops[46] = npuop(OP_REG_CNA, 0x0, CNA_DCOMP_AMOUNT15);
    ops[47] = npuop(OP_REG_CNA, 0x0, CNA_CVT_CON5);
    ops[48] = npuop(OP_REG_CNA, 0x0, CNA_PAD_CON1);

    value = ((core_desc.proc_precision as u32 & 0x7) << 8) | (core_desc.qd_en as u32 & 0x1);
    ops[49] = npuop(OP_REG_CORE, value, CORE_MISC_CFG);

    value = ((core_desc.dataout_height as u32 & 0xFFFF) << 16)
        | (core_desc.dataout_width as u32 & 0xFFFF);
    ops[50] = npuop(OP_REG_CORE, value, CORE_DATAOUT_SIZE_0);

    value = core_desc.dataout_channel as u32 & 0xFFFF;
    ops[51] = npuop(OP_REG_CORE, value, CORE_DATAOUT_SIZE_1);

    ops[52] = npuop(OP_REG_CORE, 0x0, CORE_CLIP_TRUNCATE);
    ops[53] = npuop(OP_REG_CORE, 0x0, CORE_3030);

    value = ((dpu_desc.burst_len as u32 & 0xF) << 5)
        | ((dpu_desc.conv_mode as u32 & 0x3) << 3)
        | ((dpu_desc.output_mode as u32 & 0x3) << 1)
        | (dpu_desc.flying_mode as u32 & 0x1);
    ops[54] = npuop(OP_REG_DPU, value, DPU_FEATURE_MODE_CFG);

    value = ((dpu_desc.out_precision as u32 & 0x7) << 29)
        | ((dpu_desc.in_precision as u32 & 0x7) << 26)
        | (dpu_desc.proc_precision as u32 & 0x7);
    ops[55] = npuop(OP_REG_DPU, value, DPU_DATA_FORMAT);

    ops[56] = npuop(OP_REG_DPU, 0x0, DPU_OFFSET_PEND);
    ops[57] = npuop(OP_REG_DPU, dpu_desc.dst_base_addr, DPU_DST_BASE_ADD);

    value = (dpu_desc.dst_surf_stride & 0xFFFFFFF) << 4;
    ops[58] = npuop(OP_REG_DPU, value, DPU_DST_SURF_STRIDE);

    value = dpu_desc.width as u32 & 0x1FFF;
    ops[59] = npuop(OP_REG_DPU, value, DPU_DATA_CUBE_WIDTH);

    value = dpu_desc.height as u32 & 0x1FFF;
    ops[60] = npuop(OP_REG_DPU, value, DPU_DATA_CUBE_HEIGHT);

    ops[61] = npuop(OP_REG_DPU, 0x0, DPU_DATA_CUBE_NOTCH_ADDR);

    value = ((dpu_desc.channel as u32 & 0x1FFF) << 16) | (dpu_desc.channel as u32 & 0x1FFF);
    ops[62] = npuop(OP_REG_DPU, value, DPU_DATA_CUBE_CHANNEL);

    value = ((dpu_desc.bs_relu_bypass as u32 & 0x1) << 6)
        | ((dpu_desc.bs_mul_bypass as u32 & 0x1) << 4)
        | ((dpu_desc.bs_alu_bypass as u32 & 0x1) << 1)
        | (dpu_desc.bs_bypass as u32 & 0x1);
    ops[63] = npuop(OP_REG_DPU, value, DPU_BS_CFG);

    ops[64] = npuop(OP_REG_DPU, 0x0, DPU_BS_ALU_CFG);
    ops[65] = npuop(OP_REG_DPU, 0x0, DPU_BS_MUL_CFG);
    ops[66] = npuop(OP_REG_DPU, 0x0, DPU_BS_RELUX_CMP_VALUE);

    value = ((dpu_desc.size_e_2 as u32 & 0x7) << 8)
        | ((dpu_desc.size_e_1 as u32 & 0x7) << 5)
        | ((dpu_desc.size_e_0 as u32 & 0x7) << 2)
        | ((dpu_desc.od_bypass as u32 & 0x1) << 1);
    ops[67] = npuop(OP_REG_DPU, value, DPU_BS_OW_CFG);

    ops[68] = npuop(OP_REG_DPU, 0x0, DPU_BS_OW_OP);

    value = dpu_desc.channel_wdma as u32 & 0x1FFF;
    ops[69] = npuop(OP_REG_DPU, value, DPU_WDMA_SIZE_0);

    value = ((dpu_desc.height_wdma as u32 & 0x1FFF) << 16) | (dpu_desc.width_wdma as u32 & 0x1FFF);
    ops[70] = npuop(OP_REG_DPU, value, DPU_WDMA_SIZE_1);

    value = ((dpu_desc.bn_relu_bypass as u32 & 0x1) << 6)
        | ((dpu_desc.bn_mul_bypass as u32 & 0x1) << 4)
        | ((dpu_desc.bn_alu_bypass as u32 & 0x1) << 1)
        | (dpu_desc.bn_bypass as u32 & 0x1);
    ops[71] = npuop(OP_REG_DPU, value, DPU_BN_CFG);

    ops[72] = npuop(OP_REG_DPU, 0x0, DPU_BN_ALU_CFG);
    ops[73] = npuop(OP_REG_DPU, 0x0, DPU_BN_MUL_CFG);
    ops[74] = npuop(OP_REG_DPU, 0x0, DPU_BN_RELUX_CMP_VALUE);

    value = ((dpu_desc.ew_relu_bypass as u32 & 0x1) << 9)
        | ((dpu_desc.ew_op_cvt_bypass as u32 & 0x1) << 8)
        | ((dpu_desc.ew_lut_bypass as u32 & 0x1) << 7)
        | ((dpu_desc.ew_op_bypass as u32 & 0x1) << 1)
        | (dpu_desc.ew_bypass as u32 & 0x1);
    ops[75] = npuop(OP_REG_DPU, value, DPU_EW_CFG);

    ops[76] = npuop(OP_REG_DPU, 0x0, DPU_EW_CVT_OFFSET_VALUE);
    ops[77] = npuop(OP_REG_DPU, 0x1, DPU_EW_CVT_SCALE_VALUE);
    ops[78] = npuop(OP_REG_DPU, 0x0, DPU_EW_RELUX_CMP_VALUE);
    ops[79] = npuop(OP_REG_DPU, 0x0, DPU_OUT_CVT_OFFSET);

    value =
        ((dpu_desc.fp32tofp16_en as u32 & 0x1) << 16) | (dpu_desc.out_cvt_scale as u32 & 0xFFFF);
    ops[80] = npuop(OP_REG_DPU, value, DPU_OUT_CVT_SCALE);

    ops[81] = npuop(OP_REG_DPU, 0x0, DPU_OUT_CVT_SHIFT);
    ops[82] = npuop(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_0);
    ops[83] = npuop(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_1);
    ops[84] = npuop(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_2);
    ops[85] = npuop(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_3);
    ops[86] = npuop(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_4);
    ops[87] = npuop(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_5);
    ops[88] = npuop(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_6);
    ops[89] = npuop(OP_REG_DPU, 0x0, DPU_EW_OP_VALUE_7);

    value = (dpu_desc.surf_add & 0xFFFFFFF) << 4;
    ops[90] = npuop(OP_REG_DPU, value, DPU_SURFACE_ADD);

    ops[91] = npuop(OP_REG_DPU, 0x0, DPU_40C4);
    ops[92] = npuop(OP_REG_DPU, 0x0, DPU_LUT_ACCESS_CFG);
    ops[93] = npuop(OP_REG_DPU, 0x0, DPU_LUT_ACCESS_DATA);
    ops[94] = npuop(OP_REG_DPU, 0x0, DPU_LUT_CFG);
    ops[95] = npuop(OP_REG_DPU, 0x0, DPU_LUT_INFO);
    ops[96] = npuop(OP_REG_DPU, 0x0, DPU_LUT_LE_START);
    ops[97] = npuop(OP_REG_DPU, 0x0, DPU_LUT_LE_END);
    ops[98] = npuop(OP_REG_DPU, 0x0, DPU_LUT_LO_START);
    ops[99] = npuop(OP_REG_DPU, 0x0, DPU_LUT_LO_END);
    ops[100] = npuop(OP_REG_DPU, 0x0, DPU_LUT_LE_SLOPE_SCALE);
    ops[101] = npuop(OP_REG_DPU, 0x0, DPU_LUT_LE_SLOPE_SHIFT);
    ops[102] = npuop(OP_REG_DPU, 0x0, DPU_LUT_LO_SLOPE_SCALE);
    ops[103] = npuop(OP_REG_DPU, 0x0, DPU_LUT_LO_SLOPE_SHIFT);
    ops[104] = npuop(OP_NONE, 0x0, 0x0);
    ops[105] = npuop(OP_REG_PC, 0x0, PC_REGISTER_AMOUNTS);
    ops[106] = npuop(OP_40, 0x0, 0x0);
    ops[107] = npuop(
        OP_ENABLE,
        (PC_ENABLE_DPU | PC_ENABLE_CNA | PC_ENABLE) as u32,
        PC_OPERATION_ENABLE,
    );
}

/// Generate FP16 matrix multiplication
///
/// Simplified version of matrix multiplication because:
/// a) we fail if cbuf storage is exceeded ie M,K,N get too large
/// b) because of (a) only generates a single task
///
/// task memory needs to hold at least 112 values
/// TODO: Fix a) & b)
pub fn gen_matmul_fp16(params: &mut MatmulParams) -> Result<(), i32> {
    let mut cna_desc = NpuCnaDesc::default();
    let mut core_desc = NpuCoreDesc::default();
    let mut dpu_desc = NpuDpuDesc::default();

    cna_desc.conv_mode = ConvMode::DirectConvolution as u8;
    cna_desc.in_precision = Precision::Float16 as u8;
    cna_desc.proc_precision = Precision::Float16 as u8;

    cna_desc.kernel_groups = 0;
    cna_desc.feature_grains = params.m + 1;
    cna_desc.conv_x_stride = 1;
    cna_desc.conv_y_stride = 1;

    cna_desc.datain_width = 1;
    cna_desc.datain_height = params.m;
    cna_desc.datain_channel = params.k;
    cna_desc.dataout_width = 1;
    cna_desc.dataout_height = params.m;
    cna_desc.dataout_atomics = cna_desc.dataout_width as u32 * cna_desc.dataout_height as u32;

    cna_desc.weight_width = 1;
    cna_desc.weight_height = 1;
    cna_desc.weight_kernels = params.n;
    cna_desc.weight_bytes_per_kernel = cna_desc.weight_width as u32
        * cna_desc.weight_height as u32
        * cna_desc.datain_channel as u32
        * 2; // sizeof(__fp16) = 2
    cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels as u32;

    let fd_bytes = cna_desc.datain_width as usize
        * cna_desc.datain_height as usize
        * cna_desc.datain_channel as usize
        * 2; // sizeof(__fp16) = 2
    let mut fd_banks = fd_bytes / NPU_CBUF_BANK_SIZE;
    fd_banks = if (fd_bytes % NPU_CBUF_BANK_SIZE) == 0 {
        fd_banks
    } else {
        fd_banks + 1
    };

    let weight_banks = if fd_banks > NPU_CBUF_BANKS - 1 {
        return Err(-1);
    } else {
        if cna_desc.weight_bytes_per_kernel as usize <= NPU_CBUF_BANK_SIZE {
            NPU_CBUF_BANKS - fd_banks
        } else {
            return Err(-2);
        }
    };

    cna_desc.weight_bank = weight_banks as u8;
    cna_desc.data_bank = fd_banks as u8;
    cna_desc.data_entries =
        ((cna_desc.datain_width as u32 * cna_desc.datain_channel as u32) / 32) as u16;
    cna_desc.data_entries =
        if ((cna_desc.datain_width as u32 * cna_desc.datain_channel as u32) % 32) == 0 {
            cna_desc.data_entries
        } else {
            cna_desc.data_entries + 1
        };

    cna_desc.data_sign = 0x1;
    cna_desc.cvt_type = 0x1;
    cna_desc.cvt_bypass = 0x1;
    cna_desc.cvt_scale0 = 0x1;
    cna_desc.cvt_scale1 = 0x1;
    cna_desc.cvt_scale2 = 0x1;
    cna_desc.cvt_scale3 = 0x1;
    cna_desc.fc_skip_en = 0;
    cna_desc.data_offset = 0x0;
    cna_desc.pad_left = 0;
    cna_desc.pad_top = 0;
    cna_desc.feature_base_addr = params.input_dma;
    cna_desc.weight_offset = 0;
    cna_desc.weight_burst_len = 0xf;
    cna_desc.data_burst_len = 0xf;
    cna_desc.line_stride = cna_desc.datain_width as u32 * 4;

    let mut surf_stride = cna_desc.line_stride as i32 * ((cna_desc.datain_height / 4) as i32 - 1);
    surf_stride = if surf_stride < 0 {
        surf_stride + 1
    } else {
        surf_stride
    };
    cna_desc.surf_stride = surf_stride;
    cna_desc.dma_width = cna_desc.datain_width;
    cna_desc.dma_height = cna_desc.datain_height;
    cna_desc.dma_channel = cna_desc.datain_channel;
    cna_desc.decompress_addr0 = params.weights_dma;

    core_desc.proc_precision = Precision::Float16 as u8;
    core_desc.qd_en = 1;
    core_desc.dataout_height = cna_desc.dataout_height - 1;
    core_desc.dataout_width = cna_desc.dataout_width - 1;
    core_desc.dataout_channel = cna_desc.weight_kernels - 1;

    dpu_desc.burst_len = 0xf;
    dpu_desc.conv_mode = ConvMode::DirectConvolution as u8;
    dpu_desc.output_mode = 0x2;
    dpu_desc.flying_mode = 0x0;
    dpu_desc.out_precision = if params.fp32tofp16 == 0 {
        Precision::Float32 as u8
    } else {
        Precision::Float16 as u8
    };
    dpu_desc.in_precision = Precision::Float16 as u8;
    dpu_desc.proc_precision = Precision::Float16 as u8;
    dpu_desc.dst_base_addr = params.output_dma;
    dpu_desc.dst_surf_stride = cna_desc.dataout_height as u32 * cna_desc.dataout_width as u32;
    dpu_desc.width = core_desc.dataout_width;
    dpu_desc.height = core_desc.dataout_height;
    dpu_desc.channel = core_desc.dataout_channel;
    dpu_desc.bs_bypass = 1;
    dpu_desc.bs_alu_bypass = 1;
    dpu_desc.bs_mul_bypass = 1;
    dpu_desc.bs_relu_bypass = 1;
    dpu_desc.bn_bypass = 1;
    dpu_desc.bn_alu_bypass = 1;
    dpu_desc.bn_mul_bypass = 1;
    dpu_desc.bn_relu_bypass = 1;
    dpu_desc.ew_bypass = 1;
    dpu_desc.ew_op_bypass = 1;
    dpu_desc.ew_lut_bypass = 1;
    dpu_desc.ew_op_cvt_bypass = 1;
    dpu_desc.ew_relu_bypass = 1;
    dpu_desc.fp32tofp16_en = params.fp32tofp16 & 0x1;
    dpu_desc.out_cvt_scale = 1;

    if params.fp32tofp16 == 0 {
        dpu_desc.size_e_2 = 3;
        dpu_desc.size_e_1 = 3;
        dpu_desc.size_e_0 = 3;
    } else {
        dpu_desc.size_e_2 = 1;
        dpu_desc.size_e_1 = 1;
        dpu_desc.size_e_0 = 1;
    }

    dpu_desc.od_bypass = 1;
    dpu_desc.width_wdma = core_desc.dataout_width;
    dpu_desc.height_wdma = core_desc.dataout_height;
    dpu_desc.channel_wdma = core_desc.dataout_channel;
    dpu_desc.surf_add = if params.fp32tofp16 == 0 {
        dpu_desc.dst_surf_stride * 4
    } else {
        dpu_desc.dst_surf_stride * 2
    };

    // Safety: we're writing to a pointer provided by the caller
    // The caller must ensure the pointer is valid and has at least 112 u64 elements
    unsafe {
        let ops_slice = slice::from_raw_parts_mut(params.tasks, 112);
        gen_matmul_task(ops_slice, &cna_desc, &core_desc, &dpu_desc);
    }

    Ok(())
}

/// Generate INT8 matrix multiplication
///
/// Simplified version of matrix multiplication because:
/// a) we fail if cbuf storage is exceeded ie M,K,N get too large
/// b) because of (a) only generates a single task
///
/// task memory needs to hold at least 112 values
/// TODO: Fix a) & b)
pub fn gen_matmul_int8(params: &mut MatmulParams) -> Result<(), i32> {
    let mut cna_desc = NpuCnaDesc::default();
    let mut core_desc = NpuCoreDesc::default();
    let mut dpu_desc = NpuDpuDesc::default();

    cna_desc.conv_mode = ConvMode::DirectConvolution as u8;
    cna_desc.in_precision = Precision::Int8 as u8;
    cna_desc.proc_precision = Precision::Int8 as u8;

    cna_desc.kernel_groups = 0;
    cna_desc.feature_grains = params.m + 1;
    cna_desc.conv_x_stride = 1;
    cna_desc.conv_y_stride = 1;

    cna_desc.datain_width = 1;
    cna_desc.datain_height = params.m;
    cna_desc.datain_channel = params.k;
    cna_desc.dataout_width = 1;
    cna_desc.dataout_height = params.m;
    cna_desc.dataout_atomics = cna_desc.dataout_width as u32 * cna_desc.dataout_height as u32;

    cna_desc.weight_width = 1;
    cna_desc.weight_height = 1;
    cna_desc.weight_kernels = params.n;
    cna_desc.weight_bytes_per_kernel = cna_desc.weight_width as u32
        * cna_desc.weight_height as u32
        * cna_desc.datain_channel as u32
        * 1; // sizeof(int8_t) = 1
    cna_desc.weight_bytes = cna_desc.weight_bytes_per_kernel * cna_desc.weight_kernels as u32;

    let fd_bytes = cna_desc.datain_width as usize
        * cna_desc.datain_height as usize
        * cna_desc.datain_channel as usize
        * 1; // sizeof(int8_t) = 1
    let mut fd_banks = fd_bytes / NPU_CBUF_BANK_SIZE;
    fd_banks = if (fd_bytes % NPU_CBUF_BANK_SIZE) == 0 {
        fd_banks
    } else {
        fd_banks + 1
    };

    let weight_banks = if fd_banks > NPU_CBUF_BANKS - 1 {
        return Err(-1);
    } else {
        if cna_desc.weight_bytes_per_kernel as usize <= NPU_CBUF_BANK_SIZE {
            NPU_CBUF_BANKS - fd_banks
        } else {
            return Err(-2);
        }
    };

    cna_desc.weight_bank = weight_banks as u8;
    cna_desc.data_bank = fd_banks as u8;
    cna_desc.data_entries =
        ((cna_desc.datain_width as u32 * cna_desc.datain_channel as u32) / 64) as u16;
    cna_desc.data_entries =
        if ((cna_desc.datain_width as u32 * cna_desc.datain_channel as u32) % 64) == 0 {
            cna_desc.data_entries
        } else {
            cna_desc.data_entries + 1
        };

    cna_desc.data_sign = 0x1;
    cna_desc.cvt_type = 0x1;
    cna_desc.cvt_bypass = 0x1;
    cna_desc.cvt_scale0 = 0x1;
    cna_desc.cvt_scale1 = 0x1;
    cna_desc.cvt_scale2 = 0x1;
    cna_desc.cvt_scale3 = 0x1;
    cna_desc.fc_skip_en = 0;
    cna_desc.data_offset = 0x0;
    cna_desc.pad_left = 0;
    cna_desc.pad_top = 0;
    cna_desc.feature_base_addr = params.input_dma;
    cna_desc.weight_offset = 0;
    cna_desc.weight_burst_len = 0xf;
    cna_desc.data_burst_len = 0xf;
    cna_desc.line_stride = cna_desc.datain_width as u32 * 4;

    let mut surf_stride = cna_desc.line_stride as i32 * ((cna_desc.datain_height / 4) as i32 - 1);
    surf_stride = if surf_stride < 0 {
        surf_stride + 1
    } else {
        surf_stride
    };
    cna_desc.surf_stride = surf_stride;
    cna_desc.dma_width = cna_desc.datain_width;
    cna_desc.dma_height = cna_desc.datain_height;
    cna_desc.dma_channel = cna_desc.datain_channel;
    cna_desc.decompress_addr0 = params.weights_dma;

    core_desc.proc_precision = Precision::Int8 as u8;
    core_desc.qd_en = 0;
    core_desc.dataout_height = cna_desc.dataout_height - 1;
    core_desc.dataout_width = cna_desc.dataout_width - 1;
    core_desc.dataout_channel = cna_desc.weight_kernels - 1;

    dpu_desc.burst_len = 0xf;
    dpu_desc.conv_mode = ConvMode::DirectConvolution as u8;
    dpu_desc.output_mode = 0x2;
    dpu_desc.flying_mode = 0x0;
    dpu_desc.out_precision = Precision::Int32 as u8;
    dpu_desc.in_precision = Precision::Int8 as u8;
    dpu_desc.proc_precision = Precision::Int8 as u8;
    dpu_desc.dst_base_addr = params.output_dma;
    dpu_desc.dst_surf_stride = cna_desc.dataout_height as u32 * cna_desc.dataout_width as u32;
    dpu_desc.width = core_desc.dataout_width;
    dpu_desc.height = core_desc.dataout_height;
    dpu_desc.channel = core_desc.dataout_channel;
    dpu_desc.bs_bypass = 1;
    dpu_desc.bs_alu_bypass = 1;
    dpu_desc.bs_mul_bypass = 1;
    dpu_desc.bs_relu_bypass = 1;
    dpu_desc.bn_bypass = 1;
    dpu_desc.bn_alu_bypass = 1;
    dpu_desc.bn_mul_bypass = 1;
    dpu_desc.bn_relu_bypass = 1;
    dpu_desc.ew_bypass = 1;
    dpu_desc.ew_op_bypass = 1;
    dpu_desc.ew_lut_bypass = 1;
    dpu_desc.ew_op_cvt_bypass = 1;
    dpu_desc.ew_relu_bypass = 1;
    dpu_desc.fp32tofp16_en = 0;
    dpu_desc.out_cvt_scale = 1;
    dpu_desc.size_e_2 = 7;
    dpu_desc.size_e_1 = 7;
    dpu_desc.size_e_0 = 7;
    dpu_desc.od_bypass = 1;
    dpu_desc.width_wdma = core_desc.dataout_width;
    dpu_desc.height_wdma = core_desc.dataout_height;
    dpu_desc.channel_wdma = core_desc.dataout_channel;
    dpu_desc.surf_add = dpu_desc.dst_surf_stride * 8;

    // Safety: we're writing to a pointer provided by the caller
    // The caller must ensure the pointer is valid and has at least 112 u64 elements
    unsafe {
        let ops_slice = slice::from_raw_parts_mut(params.tasks, 112);
        gen_matmul_task(ops_slice, &cna_desc, &core_desc, &dpu_desc);
    }

    Ok(())
}

/// Calculate feature data position
pub fn feature_data(_c: i32, h: i32, w: i32, c2: i32, c_val: i32, h_val: i32, w_val: i32) -> i32 {
    let plane = (c_val - 1) / c2;
    let src = plane * h * w * c2;
    let offset = (c_val - 1) % c2;
    let pos = src + c2 * ((h_val - 1) * w + (w_val - 1)) + offset;
    pos
}

/// Calculate weight position for FP16
pub fn weight_fp16(c: i32, k: i32, c_val: i32) -> i32 {
    let kpg = (k - 1) / 16;
    let cpg = (c_val - 1) / 32;
    let mut dst = (cpg * 32) * 16 + kpg * 16 * c;
    dst = dst + ((c_val - 1) % 32) + (((k - 1) % 16) * 32);
    dst
}

/// Calculate weight position for INT8
pub fn weight_int8(c: i32, k: i32, c_val: i32) -> i32 {
    let kpg = (k - 1) / 32;
    let cpg = (c_val - 1) / 32;
    let mut dst = (cpg * 32) * 32 + kpg * 32 * c;
    dst = dst + ((c_val - 1) % 32) + (((k - 1) % 32) * 32);
    dst
}
