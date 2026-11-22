// Decompile only the requested ARM or Thumb function entry points.
// @category SonyModelAnalysis
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.lang.Register;
import ghidra.program.model.listing.Function;
import java.math.BigInteger;

public class DumpSonyFunctions extends GhidraScript {
    @Override
    public void run() throws Exception {
        Register thumb = currentProgram.getRegister("TMode");
        DecompInterface decompiler = new DecompInterface();
        try {
            decompiler.openProgram(currentProgram);
            for (String arg : getScriptArgs()) {
                long operand = Long.decode(arg);
                Address entry = toAddr(operand & ~1L);
                if (getInstructionAt(entry) == null) {
                    if (thumb != null) {
                        currentProgram.getProgramContext().setValue(
                            thumb, entry, entry, BigInteger.valueOf(operand & 1L));
                    }
                    disassemble(entry);
                }
                Function function = getFunctionAt(entry);
                if (function == null) {
                    function = createFunction(entry, null);
                }
                if (function == null) {
                    println("Cannot recover a function at " + entry);
                    continue;
                }
                DecompileResults result = decompiler.decompileFunction(function, 25, monitor);
                println("Function " + entry);
                if (result.decompileCompleted()) {
                    println(result.getDecompiledFunction().getC());
                } else {
                    println(result.getErrorMessage());
                }
            }
        } finally {
            decompiler.dispose();
        }
    }
}
