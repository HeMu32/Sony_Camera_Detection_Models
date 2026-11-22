// Report containing function boundaries and direct call relationships.
// Usage: -postScript ReportSonyFunctionBounds.java <outFile> <address,...>
// @category SonyNrAnalysis
import ghidra.app.script.GhidraScript;
import ghidra.program.model.address.Address;
import ghidra.program.model.listing.Function;
import ghidra.program.model.symbol.Reference;
import java.io.File;
import java.io.FileWriter;
import java.io.PrintWriter;
import java.util.LinkedHashSet;
import java.util.Set;

public class ReportSonyFunctionBounds extends GhidraScript {
    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        if (args.length < 2) {
            throw new IllegalArgumentException("usage: outFile address,...");
        }
        try (PrintWriter out = new PrintWriter(new FileWriter(new File(args[0])))) {
            out.println("target,start,end,size,name,callers");
            for (String item : args[1].split(",")) {
                long target = Long.decode(item);
                Address address = toAddr(target & ~1L);
                Function function = getFunctionContaining(address);
                if (function == null) {
                    out.printf("0x%x,,,,,,%n", target);
                    continue;
                }
                Set<String> callers = new LinkedHashSet<>();
                for (Reference ref : getReferencesTo(function.getEntryPoint())) {
                    Function caller = getFunctionContaining(ref.getFromAddress());
                    if (caller != null) {
                        callers.add(caller.getEntryPoint() + ":" + caller.getName());
                    }
                }
                long start = function.getEntryPoint().getOffset();
                long end = function.getBody().getMaxAddress().getOffset();
                out.printf("0x%x,0x%x,0x%x,0x%x,%s,%s,%s%n",
                        target, start, end, end - start + 1,
                        function.getName(), String.join(";", callers),
                        function.getSignature());
            }
        }
        println("Wrote function bounds: " + args[0]);
    }
}
