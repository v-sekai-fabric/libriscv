#include <array>
#include "machine.hpp"
#include "decoder_cache.hpp"
#include "internal_common.hpp"
#include "instruction_counter.hpp"
#include "threaded_bytecodes.hpp"
#include "rv32i_instr.hpp"
#include "rvfd.hpp"
#ifdef RISCV_EXT_COMPRESSED
#include "rvc.hpp"
#endif
#ifdef RISCV_EXT_VECTOR
#include "rvv.hpp"
#endif

#define MUNUSED  [[maybe_unused]]
#define DISPATCH_MODE_MSVC_THREADED
#define INSTRUCTION(bytecode, name) 	template <int W> 	static bool name(DecoderData<W>*& d, MUNUSED DecodedExecuteSegment<W>*& exec, MUNUSED CPU<W>& cpu, MUNUSED address_type<W>& pc, MUNUSED InstrCounter& counter)
#define addr_t  address_type<W>
#define saddr_t signed_address_type<W>
#define XLEN    (8 * W)

#define VIEW_INSTR() 	auto instr = *(rv32i_instruction *)&d->instr;
#define VIEW_INSTR_AS(name, x) 	auto&& name = *(x *)&d->instr;
#define EXECUTE_INSTR() 	computed_opcode<W>()[d->get_bytecode()](d, exec, cpu, pc, counter)
#define EXECUTE_CURRENT() 	return true;
#define NEXT_INSTR()                   	d += (compressed_enabled ? 2 : 1); 	EXECUTE_CURRENT()
#define NEXT_C_INSTR() 	d += 1;            	EXECUTE_CURRENT()

#define RETURN_VALUES()   \
	pc
#define UNUSED_FUNCTION() \
	cpu.trigger_exception(ILLEGAL_OPCODE);

#define BEGIN_BLOCK()                               \
	pc += d->block_bytes();                         \
	counter.increment_counter(d->instruction_count());
#define NEXT_BLOCK(len, OF)              \
	pc += len;                           \
	d += len >> DecoderData<W>::SHIFT;  \
	if constexpr (OF) {                  \
		OVERFLOW_CHECK();                \
	}									 \
	if constexpr (FUZZING) /* Give OOB-aid to ASAN */          \
	d = &exec->decoder_cache()[pc >> DecoderData<W>::SHIFT];  \
	BEGIN_BLOCK()                        \
	EXECUTE_CURRENT()

#define QUICK_EXEC_CHECK()                                              \
	if (UNLIKELY(!(pc >= exec->exec_begin() && pc < exec->exec_end()))) \
		return next_execute_segment<W>(d, exec, cpu, pc, counter);

#define UNCHECKED_JUMP()                                       \
	QUICK_EXEC_CHECK()                                         \
	d = &exec->decoder_cache()[pc >> DecoderData<W>::SHIFT];  \
	BEGIN_BLOCK()                                              \
	EXECUTE_CURRENT()
#define OVERFLOW_CHECK()                            \
	if (UNLIKELY(counter.overflowed())) \
		return RETURN_VALUES();

#define PERFORM_BRANCH()                \
	if constexpr (VERBOSE_JUMPS) {      \
		printf("Branch from 0x%lX to 0x%lX\n", \
			pc, pc + fi.signed_imm());  \
	}                                   \
	pc += fi.signed_imm();              \
	d += fi.signed_imm() >> DecoderData<W>::SHIFT; \
	OVERFLOW_CHECK()                    \
	BEGIN_BLOCK()                       \
	EXECUTE_CURRENT()

#define PERFORM_FORWARD_BRANCH()        \
	if constexpr (VERBOSE_JUMPS) {      \
		printf("Fwd. Branch from 0x%lX to 0x%lX\n", \
			pc, pc + fi.signed_imm());  \
	}                                   \
	NEXT_BLOCK(fi.signed_imm(), false)

#define OVERFLOW_CHECKED_JUMP() \
	OVERFLOW_CHECK(); \
	UNCHECKED_JUMP();


namespace riscv
{
	static constexpr bool VERBOSE_JUMPS = riscv::verbose_branches_enabled;
#ifdef FUZZING_BUILD_MODE_UNSAFE_FOR_PRODUCTION
	static constexpr bool FUZZING = true;
#else
	static constexpr bool FUZZING = false;
#endif

	template <int W> static inline
	DecodedExecuteSegment<W>* resolve_execute_segment(CPU<W>& cpu, address_type<W>& pc)
	{
		// Change execute segment
		auto results = cpu.next_execute_segment(pc);
		// Restore PC
		pc = results.pc;
		return results.exec;
	}

	template <int W>
	using TcoRet = bool;

	template <int W>
	using DecoderFunc = bool(*)(DecoderData<W>*&, DecodedExecuteSegment<W>*&, CPU<W>&, address_type<W>&, InstrCounter&);
	template <int W>
	static const std::array<DecoderFunc<W>, BYTECODES_MAX>& computed_opcode();

#define DECODER()   (*d)
#define CPU()       cpu
#define REG(x)      cpu.reg(x)
#define REGISTERS() cpu.registers()
#define VECTORS()   cpu.registers().rvv()
#define MACHINE()   cpu.machine()


#include "bytecode_impl.cpp"

	INSTRUCTION(RV32I_BC_STOP, rv32i_stop)
	{
		(void) d;
		pc += 4; // Complete STOP instruction
		counter.stop();
		return RETURN_VALUES();
	}

	INSTRUCTION(RV32I_BC_SYSCALL, rv32i_syscall)
	{
		// Make the current PC visible
		cpu.registers().pc = pc;
		// Make the instruction counter(s) visible
		counter.apply(MACHINE());
		// Invoke system call
		cpu.machine().system_call(cpu.reg(REG_ECALL));
		// Restore max counter
		counter.retrieve_counters(MACHINE());
		// Clone-like system calls can change PC
		if (UNLIKELY(pc != cpu.registers().pc))
		{
			pc = cpu.registers().pc;
			QUICK_EXEC_CHECK();
			d = &exec->decoder_cache()[pc >> DecoderData<W>::SHIFT];
		}
		NEXT_BLOCK(4, true);
	}

#ifdef RISCV_BINARY_TRANSLATION
	INSTRUCTION(RV32I_BC_TRANSLATOR, translated_function) {
		VIEW_INSTR();
		auto new_values = 
			exec->mapping_at(instr.whole)(CPU(), counter.value()-1, counter.max(), pc);
		counter.set_counters(new_values.counter, new_values.max_counter);
		if (new_values.max_counter == 0) {
#ifdef RISCV_LIBTCC
			// We need to check if we have a current exception
			if (UNLIKELY(CPU().has_current_exception())) {
				const auto except = CPU().current_exception();
				CPU().clear_current_exception();
				std::rethrow_exception(except);
			}
#endif
			return RETURN_VALUES();
		}
		pc = REGISTERS().pc;
		OVERFLOW_CHECK();
		UNCHECKED_JUMP();
	}
#endif

	INSTRUCTION(RV32I_BC_SYSTEM, rv32i_system) {
		VIEW_INSTR();
		// Make the current PC visible
		cpu.registers().pc = pc;
		// Make the instruction counter visible
		counter.apply(MACHINE());
		// Invoke SYSTEM
		cpu.machine().system(instr);
		// Restore counters
		counter.retrieve_max_counter(MACHINE());
		if (UNLIKELY(pc != cpu.registers().pc))
		{
			pc = cpu.registers().pc;
			QUICK_EXEC_CHECK();
			d = &exec->decoder_cache()[pc >> DecoderData<W>::SHIFT];
		}
		// Overflow-check, next block
		NEXT_BLOCK(4, true);
	}

	INSTRUCTION(0, next_execute_segment) {
		// A helper function to change execute segment
		exec = resolve_execute_segment<W>(cpu, pc);
		d = &exec->decoder_cache()[pc >> DecoderData<W>::SHIFT];
		BEGIN_BLOCK();
		EXECUTE_CURRENT();
	}

	INSTRUCTION(RV32I_BC_INVALID, execute_invalid)
	{
		// Calculate the current PC (mid block)
		pc = (d - exec->decoder_cache()) << DecoderData<W>::SHIFT;
		// Check if the instruction is still invalid
		bool stale = false;
		try {
			if (d->instr == 0 && MACHINE().memory.template read<uint16_t>(pc) != 0) {
				exec->set_stale(true);
				stale = true;
			}
		} catch (...) {}
		if (stale) {
			exec = resolve_execute_segment<W>(cpu, pc);
			d = &exec->decoder_cache()[pc >> DecoderData<W>::SHIFT];
			NEXT_BLOCK(0, true);
		}
		cpu.registers().pc = pc;
		cpu.trigger_exception(ILLEGAL_OPCODE, d->instr);
	}

	template <int W>
	static const std::array<DecoderFunc<W>, BYTECODES_MAX>& computed_opcode()
	{
		static const std::array<DecoderFunc<W>, BYTECODES_MAX> table = [] {
			std::array<DecoderFunc<W>, BYTECODES_MAX> t{};
			for (auto& e : t) e = execute_invalid<W>;
			t[RV32I_BC_INVALID] = execute_invalid<W>;
			t[RV32I_BC_ADDI] = rv32i_addi<W>;
			t[RV32I_BC_LI] = rv32i_li<W>;
			t[RV32I_BC_MV] = rv32i_mv<W>;
			t[RV32I_BC_SLLI] = rv32i_slli<W>;
			t[RV32I_BC_SLTI] = rv32i_slti<W>;
			t[RV32I_BC_SLTIU] = rv32i_sltiu<W>;
			t[RV32I_BC_XORI] = rv32i_xori<W>;
			t[RV32I_BC_SRLI] = rv32i_srli<W>;
			t[RV32I_BC_SRAI] = rv32i_srai<W>;
			t[RV32I_BC_ORI] = rv32i_ori<W>;
			t[RV32I_BC_ANDI] = rv32i_andi<W>;
			t[RV32I_BC_LUI] = rv32i_lui<W>;
			t[RV32I_BC_AUIPC] = rv32i_auipc<W>;
			t[RV32I_BC_LDB] = rv32i_ldb<W>;
			t[RV32I_BC_LDBU] = rv32i_ldbu<W>;
			t[RV32I_BC_LDH] = rv32i_ldh<W>;
			t[RV32I_BC_LDHU] = rv32i_ldhu<W>;
			t[RV32I_BC_LDW] = rv32i_ldw<W>;
			t[RV32I_BC_STB] = rv32i_stb<W>;
			t[RV32I_BC_STH] = rv32i_sth<W>;
			t[RV32I_BC_STW] = rv32i_stw<W>;
			#ifdef RISCV_64I
			t[RV32I_BC_LDWU] = rv32i_ldwu<W>;
			t[RV32I_BC_LDD] = rv32i_ldd<W>;
			t[RV32I_BC_STD] = rv32i_std<W>;
			#endif
			t[RV32I_BC_BEQ] = rv32i_beq<W>;
			t[RV32I_BC_BNE] = rv32i_bne<W>;
			t[RV32I_BC_BLT] = rv32i_blt<W>;
			t[RV32I_BC_BGE] = rv32i_bge<W>;
			t[RV32I_BC_BLTU] = rv32i_bltu<W>;
			t[RV32I_BC_BGEU] = rv32i_bgeu<W>;
			t[RV32I_BC_BEQ_FW] = rv32i_beq_fw<W>;
			t[RV32I_BC_BNE_FW] = rv32i_bne_fw<W>;
			t[RV32I_BC_JAL] = rv32i_jal<W>;
			t[RV32I_BC_JALR] = rv32i_jalr<W>;
			t[RV32I_BC_FAST_JAL] = rv32i_fast_jal<W>;
			t[RV32I_BC_FAST_CALL] = rv32i_fast_call<W>;
			t[RV32I_BC_OP_ADD] = rv32i_op_add<W>;
			t[RV32I_BC_OP_SUB] = rv32i_op_sub<W>;
			t[RV32I_BC_OP_SLL] = rv32i_op_sll<W>;
			t[RV32I_BC_OP_SLT] = rv32i_op_slt<W>;
			t[RV32I_BC_OP_SLTU] = rv32i_op_sltu<W>;
			t[RV32I_BC_OP_XOR] = rv32i_op_xor<W>;
			t[RV32I_BC_OP_SRL] = rv32i_op_srl<W>;
			t[RV32I_BC_OP_OR] = rv32i_op_or<W>;
			t[RV32I_BC_OP_AND] = rv32i_op_and<W>;
			t[RV32I_BC_OP_MUL] = rv32i_op_mul<W>;
			t[RV32I_BC_OP_DIV] = rv32i_op_div<W>;
			t[RV32I_BC_OP_DIVU] = rv32i_op_divu<W>;
			t[RV32I_BC_OP_REM] = rv32i_op_rem<W>;
			t[RV32I_BC_OP_REMU] = rv32i_op_remu<W>;
			t[RV32I_BC_OP_SRA] = rv32i_op_sra<W>;
			t[RV32I_BC_OP_ZEXT_H] = rv32i_op_zext_h<W>;
			t[RV32I_BC_OP_SH1ADD] = rv32i_op_sh1add<W>;
			t[RV32I_BC_OP_SH2ADD] = rv32i_op_sh2add<W>;
			t[RV32I_BC_OP_SH3ADD] = rv32i_op_sh3add<W>;
			t[RV32I_BC_SEXT_B] = rv32i_sext_b<W>;
			t[RV32I_BC_SEXT_H] = rv32i_sext_h<W>;
			t[RV32I_BC_BSETI] = rv32i_bseti<W>;
			t[RV32I_BC_BEXTI] = rv32i_bexti<W>;
			#ifdef RISCV_64I
			t[RV64I_BC_ADDIW] = rv64i_addiw<W>;
			t[RV64I_BC_SLLIW] = rv64i_slliw<W>;
			t[RV64I_BC_SRLIW] = rv64i_srliw<W>;
			t[RV64I_BC_SRAIW] = rv64i_sraiw<W>;
			t[RV64I_BC_OP_ADDW] = rv64i_op_addw<W>;
			t[RV64I_BC_OP_SUBW] = rv64i_op_subw<W>;
			t[RV64I_BC_OP_MULW] = rv64i_op_mulw<W>;
			t[RV64I_BC_OP_ADD_UW] = rv64i_op_add_uw<W>;
			t[RV64I_BC_OP_SH1ADD_UW] = rv64i_op_sh1add_uw<W>;
			t[RV64I_BC_OP_SH2ADD_UW] = rv64i_op_sh2add_uw<W>;
			#endif // RISCV_64I
			#ifdef RISCV_EXT_COMPRESSED
			t[RV32C_BC_ADDI] = rv32c_addi<W>;
			t[RV32C_BC_LI] = rv32c_addi<W>;
			t[RV32C_BC_MV] = rv32c_mv<W>;
			t[RV32C_BC_SLLI] = rv32c_slli<W>;
			t[RV32C_BC_BEQZ] = rv32c_beqz<W>;
			t[RV32C_BC_BNEZ] = rv32c_bnez<W>;
			t[RV32C_BC_JMP] = rv32c_jmp<W>;
			t[RV32C_BC_JR] = rv32c_jr<W>;
			t[RV32C_BC_JAL_ADDIW] = rv32c_jal_addiw<W>;
			t[RV32C_BC_JALR] = rv32c_jalr<W>;
			t[RV32C_BC_LDD] = rv32c_ldd<W>;
			t[RV32C_BC_STD] = rv32c_std<W>;
			t[RV32C_BC_LDW] = rv32c_ldw<W>;
			t[RV32C_BC_STW] = rv32c_stw<W>;
			t[RV32C_BC_SRLI] = rv32c_srli<W>;
			t[RV32C_BC_ANDI] = rv32c_andi<W>;
			t[RV32C_BC_ADD] = rv32c_add<W>;
			t[RV32C_BC_XOR] = rv32c_xor<W>;
			t[RV32C_BC_OR] = rv32c_or<W>;
			t[RV32C_BC_FUNCTION] = rv32c_func<W>;
			#endif
			t[RV32I_BC_SYSCALL] = rv32i_syscall<W>;
			t[RV32I_BC_STOP] = rv32i_stop<W>;
			t[RV32F_BC_FLW] = rv32i_flw<W>;
			t[RV32F_BC_FLD] = rv32i_fld<W>;
			t[RV32F_BC_FSW] = rv32i_fsw<W>;
			t[RV32F_BC_FSD] = rv32i_fsd<W>;
			t[RV32F_BC_FADD] = rv32f_fadd<W>;
			t[RV32F_BC_FSUB] = rv32f_fsub<W>;
			t[RV32F_BC_FMUL] = rv32f_fmul<W>;
			t[RV32F_BC_FDIV] = rv32f_fdiv<W>;
			t[RV32F_BC_FMADD] = rv32f_fmadd<W>;
			#ifdef RISCV_EXT_VECTOR
			t[RV32V_BC_VLE32] = rv32v_vle32<W>;
			t[RV32V_BC_VSE32] = rv32v_vse32<W>;
			t[RV32V_BC_VFADD_VV] = rv32v_vfadd_vv<W>;
			t[RV32V_BC_VFMUL_VF] = rv32v_vfmul_vf<W>;
			#endif
			t[RV32I_BC_FUNCTION] = execute_decoded_function<W>;
			t[RV32I_BC_FUNCBLOCK] = execute_function_block<W>;
			#ifdef RISCV_BINARY_TRANSLATION
			t[RV32I_BC_TRANSLATOR] = translated_function<W>;
			#endif
			t[RV32I_BC_LIVEPATCH] = execute_livepatch<W>;
			t[RV32I_BC_SYSTEM] = rv32i_system<W>;
			return t;
		}();
		return table;
	}

	template <int W> inline RISCV_HOT_PATH()
	bool CPU<W>::simulate(address_t pc, uint64_t inscounter, uint64_t maxcounter)
	{
		InstrCounter counter{inscounter, maxcounter};

		auto* exec = this->m_exec;

		// We need an execute segment matching current PC
		if (UNLIKELY(!exec->is_within(pc)))
		{
			auto results = this->next_execute_segment(pc);
			exec = results.exec;
			pc   = results.pc;
		}

		DecoderData<W>* exec_decoder = exec->decoder_cache();
		auto* d = &exec_decoder[pc >> DecoderData<W>::SHIFT];
		auto& cpu = *this;

		BEGIN_BLOCK();

		while (EXECUTE_INSTR()) {
			// each handler advances d, pc and exec before asking for re-dispatch
		}

		cpu.registers().pc = pc;
		MACHINE().set_instruction_counter(counter.value());

		// Machine stopped normally?
		return counter.max() == 0;

	} // CPU::simulate()

	template <int W>
	void CPU<W>::simulate_inaccurate(address_t pc)
	{
		machine().set_instruction_counter(0);
		machine().set_max_instructions(UINT64_MAX);
		InstrCounter counter{0, UINT64_MAX};

		auto* exec = this->m_exec;

		// We need an execute segment matching current PC
		if (UNLIKELY(!exec->is_within(pc)))
		{
			auto results = this->next_execute_segment(pc);
			exec = results.exec;
			pc   = results.pc;
		}

		DecoderData<W>* exec_decoder = exec->decoder_cache();
		auto* d = &exec_decoder[pc >> DecoderData<W>::SHIFT];
		auto& cpu = *this;

		BEGIN_BLOCK();

		while (EXECUTE_INSTR()) {
		}

		cpu.registers().pc = pc;
	}

	INSTANTIATE_32_IF_ENABLED(CPU);
	INSTANTIATE_64_IF_ENABLED(CPU);
	INSTANTIATE_128_IF_ENABLED(CPU);
} // riscv
